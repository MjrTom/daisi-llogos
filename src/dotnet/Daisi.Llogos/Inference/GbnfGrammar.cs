using Daisi.Llogos.Tokenizer;

namespace Daisi.Llogos.Inference;

/// <summary>
/// GBNF grammar-constrained sampling, following the llama.cpp approach:
///
/// 1. Parse grammar into flat rules where each rule is a sequence of terminal elements
///    (char ranges, literals) and rule references, with alternates as separate sub-sequences.
///    Optional, Repeat, and Group constructs are desugared into helper rules at parse time.
///
/// 2. Grammar state = set of stacks. Each stack is a list of positions within rules.
///    Stacks are always pre-resolved: the top always points at a terminal element.
///    advance_stack() expands rule refs using a worklist with dedup.
///
/// 3. Token validation: for each candidate token, test character-by-character against
///    each stack. A token is rejected only if ALL stacks reject it.
///
/// 4. Performance: pre-resolving stacks means validation is just char matching,
///    no complex expansion during the hot path.
/// </summary>
public sealed class GrammarConstraint
{
    private readonly GbnfGrammar _grammar;
    private readonly BpeTokenizer _tokenizer;
    private List<GrammarStack> _stacks;
    private readonly string?[] _tokenTextCache; // Cache token ID → decoded text

    public GrammarConstraint(string grammarText, BpeTokenizer tokenizer, string rootRule = "root")
    {
        _grammar = new GbnfGrammar(grammarText);
        _tokenizer = tokenizer;
        _stacks = _grammar.GetInitialStacks(rootRule);

        // Pre-cache all token texts (avoids repeated decode calls in ApplyToLogits)
        _tokenTextCache = new string?[tokenizer.Vocabulary.Count];
        for (int i = 0; i < _tokenTextCache.Length; i++)
            _tokenTextCache[i] = tokenizer.Decode([i]);
    }

    /// <summary>
    /// Apply grammar constraints to logits: set invalid tokens to -inf.
    ///
    /// Optimization: pre-resolve stacks to get valid first chars, then only check
    /// tokens whose first char can advance the grammar. This eliminates ~99% of
    /// candidates before the expensive per-stack character-by-character check.
    /// </summary>
    public void ApplyToLogits(Span<float> logits)
    {
        if (_stacks.Count == 0) return;

        // Step 1: Resolve all stacks to terminals and collect valid first chars
        var resolvedStacks = new List<GrammarStack>();
        var allowedFirstChars = new HashSet<char>();
        foreach (var stack in _stacks)
        {
            var terminals = _grammar.AdvanceStackPublic(stack);
            resolvedStacks.AddRange(terminals);
            foreach (var t in terminals)
            {
                if (t.Count == 0) continue;
                var (ruleIdx, elemIdx) = t[^1];
                var elem = _grammar.GetRules()[ruleIdx].Elements[elemIdx];
                CollectFirstChars(elem, allowedFirstChars);
            }
        }

        // Step 2: Build candidate list using first-char filter
        var candidates = new List<GrammarCandidate>();
        int cacheLen = Math.Min(logits.Length, _tokenTextCache.Length);
        for (int i = 0; i < cacheLen; i++)
        {
            if (logits[i] == float.NegativeInfinity) continue;
            var text = _tokenTextCache[i];
            if (string.IsNullOrEmpty(text)) { logits[i] = float.NegativeInfinity; continue; }

            // First-char rejection: skip tokens whose first char can't match any terminal
            if (!allowedFirstChars.Contains(text[0]))
            {
                logits[i] = float.NegativeInfinity;
                continue;
            }

            candidates.Add(new GrammarCandidate(i, text));
        }
        // Handle any tokens beyond cache
        for (int i = cacheLen; i < logits.Length; i++)
            logits[i] = float.NegativeInfinity;

        // Step 3: Reject candidates that fail ALL resolved stacks
        var rejected = _grammar.RejectCandidates(resolvedStacks, candidates);
        foreach (var r in rejected)
            logits[r.TokenIndex] = float.NegativeInfinity;
    }

    private static void CollectFirstChars(GrammarElem elem, HashSet<char> chars)
    {
        if (elem.Ranges == null) return;
        if (elem.Type == GrammarElemType.CharNot)
        {
            // Negated: add all printable chars except excluded
            for (int c = 32; c < 127; c++)
                if (elem.MatchesChar((char)c)) chars.Add((char)c);
            for (int c = 128; c < 256; c++)
                if (elem.MatchesChar((char)c)) chars.Add((char)c);
        }
        else
        {
            foreach (var (from, to) in elem.Ranges)
                for (char c = from; c <= to && c < 256; c++)
                    chars.Add(c);
        }
    }

    /// <summary>
    /// Advance the grammar state after a token is sampled.
    /// </summary>
    public void Accept(int tokenId)
    {
        string tokenText = _tokenizer.Decode([tokenId]);
        if (string.IsNullOrEmpty(tokenText)) return;

        var newStacks = new List<GrammarStack>();
        foreach (var stack in _stacks)
        {
            var advanced = _grammar.AcceptChars(stack, tokenText);
            newStacks.AddRange(advanced);
        }
        _stacks = GbnfGrammar.Deduplicate(newStacks);
    }

    public bool IsComplete => _stacks.Any(s => s.Count == 0);
}

internal readonly record struct GrammarCandidate(int TokenIndex, string Text);

/// <summary>
/// A grammar stack: list of positions (ruleIndex, elementIndex) within the grammar.
/// Top of stack = last element. An empty stack means the grammar has completed.
/// </summary>
internal sealed class GrammarStack : List<(int ruleIdx, int elemIdx)>
{
    internal GrammarStack() { }
    internal GrammarStack(IEnumerable<(int, int)> items) : base(items) { }
    internal GrammarStack Clone() => new(this);
}

/// <summary>
/// GBNF grammar engine. Parses GBNF text, desugars Optional/Repeat/Group into flat rules,
/// and provides stack-based NFA simulation for grammar-constrained generation.
/// </summary>
internal sealed class GbnfGrammar
{
    // Each rule is a list of elements. Alternates are separate rules sharing the same name.
    // Element types: Char (match one char), CharNot (negative match), Literal (multi-char),
    //                RuleRef (push rule), Alt (marks alternate boundary), End (marks rule end).
    private readonly List<GrammarRule> _rules = [];
    private int _nextSyntheticId;

    internal IReadOnlyList<GrammarRule> GetRules() => _rules;

    internal GbnfGrammar(string grammarText)
    {
        Parse(grammarText);
    }

    internal List<GrammarStack> GetInitialStacks(string rootRule)
    {
        var alts = GetRuleAlternatesByName(rootRule);
        if (alts.Count == 0) return [];

        var result = new List<GrammarStack>();
        foreach (int altIdx in alts)
        {
            var stack = new GrammarStack();
            if (_rules[altIdx].Elements.Length > 0)
                stack.Add((altIdx, 0));
            result.AddRange(AdvanceStack(stack));
        }
        return Deduplicate(result);
    }

    /// <summary>
    /// Reject candidates that are invalid in ALL stacks. Returns the rejected subset.
    /// A token is valid if ANY stack accepts it.
    /// </summary>
    internal List<GrammarCandidate> RejectCandidates(List<GrammarStack> stacks, List<GrammarCandidate> candidates)
    {
        // Start with all candidates as "rejected by every stack so far"
        var rejects = candidates;

        foreach (var stack in stacks)
        {
            if (rejects.Count == 0) break;
            rejects = RejectCandidatesForStack(stack, rejects, 0);
        }

        return rejects;
    }

    /// <summary>
    /// From the candidate list, return those rejected by this specific stack.
    /// Processes character-by-character through each candidate's text.
    /// </summary>
    private List<GrammarCandidate> RejectCandidatesForStack(
        GrammarStack stack, List<GrammarCandidate> candidates, int charOffset)
    {
        if (stack.Count == 0)
        {
            // Empty stack = grammar complete. Only accept empty remaining text.
            return candidates.Where(c => charOffset < c.Text.Length).ToList();
        }

        var (ruleIdx, elemIdx) = stack[^1];
        var rule = _rules[ruleIdx];
        var elem = rule.Elements[elemIdx];

        var accepted = new List<GrammarCandidate>();
        var rejected = new List<GrammarCandidate>();

        foreach (var cand in candidates)
        {
            if (charOffset >= cand.Text.Length)
            {
                // Token fully consumed — accepted by this stack (partial match is OK)
                accepted.Add(cand);
                continue;
            }

            char c = cand.Text[charOffset];

            if (elem.MatchesChar(c))
                accepted.Add(cand);
            else
                rejected.Add(cand);
        }

        if (accepted.Count == 0) return candidates; // All rejected

        // Advance stack past the matched char element and recurse for next char
        var advancedStack = stack.Clone();
        advancedStack.RemoveAt(advancedStack.Count - 1);

        // Push the next element in the same rule (if any)
        if (elemIdx + 1 < rule.Elements.Length)
            advancedStack.Add((ruleIdx, elemIdx + 1));

        // Expand any rule refs at the new top
        var newStacks = AdvanceStack(advancedStack);

        // Recursively check remaining characters against all new stacks
        var stillRejected = accepted; // Start assuming all accepted tokens might be rejected at next char
        foreach (var ns in newStacks)
        {
            if (stillRejected.Count == 0) break;
            stillRejected = RejectCandidatesForStack(ns, stillRejected, charOffset + 1);
        }

        // Combine: tokens rejected at this char + tokens accepted at this char but rejected at later chars
        rejected.AddRange(stillRejected);
        return rejected;
    }

    /// <summary>
    /// Accept a string of characters through a stack, returning all valid resulting stacks.
    /// Used by GrammarConstraint.Accept after a token is sampled.
    /// </summary>
    internal List<GrammarStack> AcceptChars(GrammarStack stack, string text)
    {
        // Ensure initial stack is resolved to terminals
        var current = AdvanceStack(stack);

        foreach (char c in text)
        {
            var next = new List<GrammarStack>();
            foreach (var s in current)
            {
                if (s.Count == 0) continue;
                var (ruleIdx, elemIdx) = s[^1];
                var elem = _rules[ruleIdx].Elements[elemIdx];

                if (!elem.MatchesChar(c)) continue;

                var advanced = s.Clone();
                advanced.RemoveAt(advanced.Count - 1);
                if (elemIdx + 1 < _rules[ruleIdx].Elements.Length)
                    advanced.Add((ruleIdx, elemIdx + 1));

                // Resolve the advanced stack so all tops are terminals for the next iteration
                next.AddRange(AdvanceStack(advanced));
            }
            if (next.Count == 0) return [];
            current = Deduplicate(next);
        }
        return current;
    }

    /// <summary>
    /// Expand rule references at the top of a stack until all resulting stacks
    /// point at terminal elements (char matchers) or are empty (complete).
    /// Uses worklist with dedup to handle recursive rules.
    /// </summary>
    internal List<GrammarStack> AdvanceStackPublic(GrammarStack stack) => AdvanceStack(stack);

    private List<GrammarStack> AdvanceStack(GrammarStack stack)
    {
        var result = new List<GrammarStack>();
        var todo = new List<GrammarStack> { stack };
        var seen = new HashSet<string>();

        while (todo.Count > 0)
        {
            var cur = todo[^1];
            todo.RemoveAt(todo.Count - 1);

            // Dedup by stack signature
            var key = StackKey(cur);
            if (!seen.Add(key)) continue;

            if (cur.Count == 0)
            {
                // Empty stack = complete parse path
                result.Add(cur);
                continue;
            }

            var (ruleIdx, elemIdx) = cur[^1];
            var rule = _rules[ruleIdx];

            // Skip past any end-of-sequence markers
            if (elemIdx >= rule.Elements.Length)
            {
                // Rule exhausted — pop frame and continue
                var popped = cur.Clone();
                popped.RemoveAt(popped.Count - 1);
                todo.Add(popped);
                continue;
            }

            var elem = rule.Elements[elemIdx];

            if (elem.Type == GrammarElemType.RuleRef)
            {
                // Expand: pop the rule ref, push continuation, then push each alternate of the referenced rule
                var alts = GetRuleAlternatesByName(elem.RuleName!);

                foreach (int altRuleIdx in alts)
                {
                    var ns = cur.Clone();
                    ns.RemoveAt(ns.Count - 1);
                    // Push continuation (next element after the rule ref)
                    if (elemIdx + 1 < rule.Elements.Length)
                        ns.Add((ruleIdx, elemIdx + 1));
                    // Push the alternate rule
                    if (_rules[altRuleIdx].Elements.Length > 0)
                        ns.Add((altRuleIdx, 0));
                    todo.Add(ns);
                }
            }
            else
            {
                // Terminal element — stack is ready
                result.Add(cur);
            }
        }

        return Deduplicate(result);
    }

    /// <summary>Get all alternate rule indices for a rule name.</summary>
    private List<int> GetRuleAlternatesByName(string name)
    {
        var alts = new List<int>();
        for (int i = 0; i < _rules.Count; i++)
            if (_rules[i].Name == name) alts.Add(i);
        return alts;
    }

    private static string StackKey(GrammarStack stack)
    {
        if (stack.Count == 0) return "[]";
        var sb = new System.Text.StringBuilder(stack.Count * 8);
        for (int i = 0; i < stack.Count; i++)
        {
            if (i > 0) sb.Append('|');
            sb.Append(stack[i].ruleIdx);
            sb.Append(':');
            sb.Append(stack[i].elemIdx);
        }
        return sb.ToString();
    }

    internal static List<GrammarStack> Deduplicate(List<GrammarStack> stacks)
    {
        var seen = new HashSet<string>();
        var result = new List<GrammarStack>();
        foreach (var s in stacks)
        {
            var key = StackKey(s);
            if (seen.Add(key)) result.Add(s);
        }
        return result;
    }

    // ── Parser ──────────────────────────────────────────────────────────────

    private void Parse(string text)
    {
        // First pass: parse into intermediate AST
        var rawRules = new Dictionary<string, GbnfElement[]>();
        var lines = text.Split('\n');
        string? currentName = null;
        string currentExpr = "";

        foreach (var rawLine in lines)
        {
            var line = rawLine.Trim();
            int commentIdx = FindComment(line);
            if (commentIdx >= 0) line = line[..commentIdx].TrimEnd();
            if (line.Length == 0) continue;

            int defIdx = line.IndexOf("::=", StringComparison.Ordinal);
            if (defIdx >= 0)
            {
                if (currentName != null)
                    rawRules[currentName] = ParseElements(currentExpr.Trim());
                currentName = line[..defIdx].Trim();
                currentExpr = line[(defIdx + 3)..];
            }
            else if (currentName != null)
                currentExpr += " " + line;
        }
        if (currentName != null)
            rawRules[currentName] = ParseElements(currentExpr.Trim());

        // Second pass: desugar AST into flat rules
        foreach (var (name, elements) in rawRules)
            DesugarRule(name, elements);
    }

    /// <summary>
    /// Convert an AST element array into one or more flat GrammarRules.
    /// Desugars Optional, Repeat, Group into synthetic rules with alternates.
    /// </summary>
    private void DesugarRule(string name, GbnfElement[] elements)
    {
        if (elements.Length == 1 && elements[0] is GbnfAlternationElement alt)
        {
            // Top-level alternation: create one rule per branch
            foreach (var branch in alt.Branches)
            {
                var flatElems = new List<GrammarElem>();
                FlattenElements(branch, flatElems);
                AddRule(name, flatElems);
            }
        }
        else
        {
            var flatElems = new List<GrammarElem>();
            FlattenElements(elements, flatElems);
            AddRule(name, flatElems);
        }
    }

    /// <summary>
    /// Flatten AST elements into a flat list of GrammarElems, creating synthetic rules
    /// for Optional, Repeat, and Group constructs.
    /// </summary>
    private void FlattenElements(GbnfElement[] elements, List<GrammarElem> output)
    {
        foreach (var elem in elements)
        {
            switch (elem)
            {
                case GbnfLiteralElement lit:
                    // Expand multi-char literal into individual char matchers
                    foreach (char c in lit.Text)
                        output.Add(GrammarElem.Char(c));
                    break;

                case GbnfCharClassElement cc:
                    output.Add(GrammarElem.CharClass(cc));
                    break;

                case GbnfRuleRefElement rref:
                    output.Add(GrammarElem.RuleRef(rref.Name));
                    break;

                case GbnfAlternationElement altElem:
                    // Create synthetic rule for inline alternation
                    var synName = NextSyntheticName();
                    foreach (var branch in altElem.Branches)
                    {
                        var branchElems = new List<GrammarElem>();
                        FlattenElements(branch, branchElems);
                        AddRule(synName, branchElems);
                    }
                    output.Add(GrammarElem.RuleRef(synName));
                    break;

                case GbnfOptionalElement opt:
                    // x? → synthetic rule: x | ε
                    var optName = NextSyntheticName();
                    var optInner = opt.Inner is GbnfGroupElement og ? og.Elements : [opt.Inner];
                    var innerElems = new List<GrammarElem>();
                    FlattenElements(optInner, innerElems);
                    AddRule(optName, innerElems); // x branch
                    AddRule(optName, []); // ε branch (empty)
                    output.Add(GrammarElem.RuleRef(optName));
                    break;

                case GbnfRepeatElement rep:
                    // x* → synthetic rule: _rep ::= x _rep | ε
                    // x+ → synthetic rule: _rep ::= x _rep | x
                    var repName = NextSyntheticName();
                    GbnfElement[] repInner;
                    if (rep.Inner is GbnfGroupElement rg)
                        repInner = rg.Elements;
                    else if (rep.Inner is GbnfAlternationElement)
                        repInner = [rep.Inner]; // Keep alternation as single element for FlattenElements to desugar
                    else
                        repInner = [rep.Inner];

                    // Recursive branch: inner + self-ref
                    var recElems = new List<GrammarElem>();
                    FlattenElements(repInner, recElems);
                    recElems.Add(GrammarElem.RuleRef(repName));
                    AddRule(repName, recElems);

                    if (rep.MinCount == 0)
                    {
                        // ε branch (zero occurrences)
                        AddRule(repName, []);
                    }
                    else
                    {
                        // Base branch: just inner (one occurrence)
                        var baseElems = new List<GrammarElem>();
                        FlattenElements(repInner, baseElems);
                        AddRule(repName, baseElems);
                    }
                    output.Add(GrammarElem.RuleRef(repName));
                    break;

                case GbnfGroupElement grp:
                    // Inline group: just flatten its elements
                    FlattenElements(grp.Elements, output);
                    break;
            }
        }
    }

    private void AddRule(string name, List<GrammarElem> elements)
    {
        _rules.Add(new GrammarRule(name, elements.ToArray()));
    }

    private string NextSyntheticName() => $"_s{_nextSyntheticId++}";

    // ── Expression parser (produces AST) ────────────────────────────────────

    private static int FindComment(string line)
    {
        bool inQuotes = false;
        for (int i = 0; i < line.Length; i++)
        {
            if (line[i] == '"' && (i == 0 || line[i - 1] != '\\')) inQuotes = !inQuotes;
            if (!inQuotes && line[i] == '#') return i;
        }
        return -1;
    }

    private GbnfElement[] ParseElements(string expr)
    {
        var alts = SplitTopLevel(expr, '|');
        if (alts.Count > 1)
            return [new GbnfAlternationElement(alts.Select(a => ParseElements(a.Trim())).ToArray())];

        var elements = new List<GbnfElement>();
        int i = 0;
        while (i < expr.Length)
        {
            while (i < expr.Length && char.IsWhiteSpace(expr[i])) i++;
            if (i >= expr.Length) break;

            GbnfElement? elem = null;

            if (expr[i] == '"') elem = ParseLiteral(expr, ref i);
            else if (expr[i] == '[') elem = ParseCharClass(expr, ref i);
            else if (expr[i] == '(')
            {
                int depth = 1, start = ++i;
                while (i < expr.Length && depth > 0) { if (expr[i] == '(') depth++; else if (expr[i] == ')') depth--; i++; }
                var inner = ParseElements(expr[start..(i - 1)]);
                elem = inner.Length == 1 ? inner[0] : new GbnfGroupElement(inner);
            }
            else if (char.IsLetterOrDigit(expr[i]) || expr[i] == '_' || expr[i] == '-')
            {
                int start = i;
                while (i < expr.Length && (char.IsLetterOrDigit(expr[i]) || expr[i] == '_' || expr[i] == '-')) i++;
                elem = new GbnfRuleRefElement(expr[start..i]);
            }
            else { i++; continue; }

            if (elem == null) continue;

            if (i < expr.Length && expr[i] == '*') { elem = new GbnfRepeatElement(elem, 0); i++; }
            else if (i < expr.Length && expr[i] == '+') { elem = new GbnfRepeatElement(elem, 1); i++; }
            else if (i < expr.Length && expr[i] == '?') { elem = new GbnfOptionalElement(elem); i++; }
            else if (i < expr.Length && expr[i] == '{')
            {
                i++; int s = i; while (i < expr.Length && expr[i] != '}') i++;
                var parts = expr[s..i].Split(',');
                elem = new GbnfRepeatElement(elem, int.Parse(parts[0].Trim()));
                i++;
            }

            elements.Add(elem);
        }
        return elements.ToArray();
    }

    private static GbnfLiteralElement ParseLiteral(string expr, ref int i)
    {
        i++;
        var sb = new System.Text.StringBuilder();
        while (i < expr.Length && expr[i] != '"')
        {
            if (expr[i] == '\\' && i + 1 < expr.Length)
            {
                i++;
                sb.Append(expr[i] switch { 'n' => '\n', 't' => '\t', 'r' => '\r', '"' => '"', '\\' => '\\', _ => expr[i] });
            }
            else sb.Append(expr[i]);
            i++;
        }
        if (i < expr.Length) i++;
        return new GbnfLiteralElement(sb.ToString());
    }

    private static GbnfCharClassElement ParseCharClass(string expr, ref int i)
    {
        i++;
        bool neg = false;
        if (i < expr.Length && expr[i] == '^') { neg = true; i++; }
        var ranges = new List<(char, char)>();
        while (i < expr.Length && expr[i] != ']')
        {
            char from = ReadCharClassChar(expr, ref i);
            if (i + 1 < expr.Length && expr[i] == '-' && expr[i + 1] != ']')
            {
                i++; // skip '-'
                char to = ReadCharClassChar(expr, ref i);
                ranges.Add((from, to));
            }
            else
            {
                ranges.Add((from, from));
            }
        }
        if (i < expr.Length) i++;
        return new GbnfCharClassElement(ranges.ToArray(), neg);
    }

    /// <summary>Read a single character from a char class, handling \x hex and \n \t \r \\ escapes.</summary>
    private static char ReadCharClassChar(string expr, ref int i)
    {
        if (i < expr.Length && expr[i] == '\\' && i + 1 < expr.Length)
        {
            i++; // skip backslash
            char esc = expr[i];
            if (esc == 'x' && i + 2 < expr.Length)
            {
                // \xHH hex escape
                var hex = expr.Substring(i + 1, 2);
                if (int.TryParse(hex, System.Globalization.NumberStyles.HexNumber, null, out int val))
                {
                    i += 3; // skip 'x' + 2 hex digits
                    return (char)val;
                }
            }
            i++; // skip escape char
            return esc switch { 'n' => '\n', 't' => '\t', 'r' => '\r', _ => esc };
        }
        return expr[i++];
    }

    private static List<string> SplitTopLevel(string expr, char sep)
    {
        var result = new List<string>();
        int parenDepth = 0; int bracketDepth = 0; bool inQ = false; int start = 0;
        for (int i = 0; i < expr.Length; i++)
        {
            // Only toggle quote state outside brackets (quotes inside [...] are literal chars, not string delimiters)
            if (bracketDepth == 0 && expr[i] == '"' && (i == 0 || expr[i - 1] != '\\')) inQ = !inQ;
            if (!inQ)
            {
                if (expr[i] == '(') parenDepth++;
                else if (expr[i] == ')') parenDepth--;
                else if (expr[i] == '[') bracketDepth++;
                else if (expr[i] == ']') bracketDepth--;
            }
            if (!inQ && parenDepth == 0 && bracketDepth == 0 && expr[i] == sep) { result.Add(expr[start..i]); start = i + 1; }
        }
        result.Add(expr[start..]);
        return result;
    }
}

// ── Flat grammar elements (runtime) ──────────────────────────────────────

internal enum GrammarElemType { Char, CharNot, RuleRef }

internal readonly struct GrammarElem
{
    public GrammarElemType Type { get; init; }
    public (char from, char to)[]? Ranges { get; init; }    // For Char/CharNot: char ranges
    public string? RuleName { get; init; }                   // For RuleRef: rule name (resolved lazily)

    public static GrammarElem Char(char c) => new() { Type = GrammarElemType.Char, Ranges = [(c, c)] };
    public static GrammarElem CharClass(GbnfCharClassElement cc) => new()
    {
        Type = cc.IsNegated ? GrammarElemType.CharNot : GrammarElemType.Char,
        Ranges = cc.Ranges
    };
    public static GrammarElem RuleRef(string name) => new()
    {
        Type = GrammarElemType.RuleRef,
        RuleName = name
    };

    public bool MatchesChar(char c)
    {
        if (Ranges == null) return false;
        bool inRange = false;
        foreach (var (f, t) in Ranges)
            if (c >= f && c <= t) { inRange = true; break; }
        return Type == GrammarElemType.CharNot ? !inRange : inRange;
    }
}

internal sealed class GrammarRule(string name, GrammarElem[] elements)
{
    public string Name => name;
    public GrammarElem[] Elements => elements;
}

// ── AST element types (parse-time only) ──────────────────────────────────

internal abstract class GbnfElement { }
internal sealed class GbnfLiteralElement(string text, int charIndex = 0) : GbnfElement
{
    public string Text => text;
    public int CharIndex => charIndex;
}
internal sealed class GbnfCharClassElement((char from, char to)[] ranges, bool negated) : GbnfElement
{
    public (char from, char to)[] Ranges => ranges;
    public bool IsNegated => negated;
    public bool Matches(char c)
    {
        bool inRange = false;
        foreach (var (f, t) in ranges) if (c >= f && c <= t) { inRange = true; break; }
        return negated ? !inRange : inRange;
    }
}
internal sealed class GbnfRuleRefElement(string name) : GbnfElement { public string Name => name; }
internal sealed class GbnfAlternationElement(GbnfElement[][] branches) : GbnfElement { public GbnfElement[][] Branches => branches; }
internal sealed class GbnfGroupElement(GbnfElement[] elements) : GbnfElement { public GbnfElement[] Elements => elements; }
internal sealed class GbnfRepeatElement(GbnfElement inner, int minCount) : GbnfElement { public GbnfElement Inner => inner; public int MinCount => minCount; }
internal sealed class GbnfOptionalElement(GbnfElement inner) : GbnfElement { public GbnfElement Inner => inner; }
