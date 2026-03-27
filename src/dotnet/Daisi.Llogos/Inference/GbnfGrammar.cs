using Daisi.Llogos.Tokenizer;

namespace Daisi.Llogos.Inference;

/// <summary>
/// GBNF grammar-constrained sampling. Parses GBNF text into rules, tracks
/// generation state, and masks invalid tokens at each sampling step.
///
/// The approach: maintain a set of valid "grammar stacks" (positions within
/// rules). For each candidate token, check if its text can advance any
/// valid stack. Tokens that can't are masked to -inf.
///
/// This is a character-level constraint: each character of a token's text
/// must match the grammar. Multi-character tokens are validated by stepping
/// through each character.
/// </summary>
public sealed class GrammarConstraint
{
    private readonly GbnfGrammar _grammar;
    private readonly BpeTokenizer _tokenizer;
    private List<GrammarStack> _stacks;

    public GrammarConstraint(string grammarText, BpeTokenizer tokenizer, string rootRule = "root")
    {
        _grammar = new GbnfGrammar(grammarText);
        _tokenizer = tokenizer;
        // Initialize with the root rule's starting positions
        _stacks = _grammar.GetInitialStacks(rootRule);
    }

    /// <summary>
    /// Apply grammar constraints to logits: set invalid tokens to -inf.
    /// Call before sampling each token.
    /// </summary>
    public void ApplyToLogits(Span<float> logits)
    {
        if (_stacks.Count == 0) return; // Grammar complete or failed

        for (int tokenId = 0; tokenId < logits.Length; tokenId++)
        {
            if (logits[tokenId] == float.NegativeInfinity) continue;

            string tokenText = _tokenizer.Decode([tokenId]);
            if (string.IsNullOrEmpty(tokenText)) continue;

            // Check if this token's text can advance any valid stack
            if (!CanAccept(tokenText))
                logits[tokenId] = float.NegativeInfinity;
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
            var advanced = _grammar.AdvanceStack(stack, tokenText);
            newStacks.AddRange(advanced);
        }
        _stacks = Deduplicate(newStacks);
    }

    /// <summary>Whether the grammar has been fully matched (generation can stop).</summary>
    public bool IsComplete => _stacks.Any(s => s.IsComplete);

    private bool CanAccept(string text)
    {
        foreach (var stack in _stacks)
        {
            var advanced = _grammar.AdvanceStack(stack, text);
            if (advanced.Count > 0) return true;
        }
        return false;
    }

    private static List<GrammarStack> Deduplicate(List<GrammarStack> stacks)
    {
        // Simple dedup by string key
        var seen = new HashSet<string>();
        var result = new List<GrammarStack>();
        foreach (var s in stacks)
        {
            var key = s.GetKey();
            if (seen.Add(key))
                result.Add(s);
        }
        return result;
    }
}

/// <summary>
/// A position within the grammar's rule tree. Tracks which rule elements
/// have been matched and what comes next.
/// </summary>
internal sealed class GrammarStack
{
    /// <summary>Stack of (rule elements, position within elements) from outer to inner.</summary>
    internal readonly List<(GbnfElement[] elements, int pos)> Frames;

    internal GrammarStack() { Frames = new(); }
    internal GrammarStack(List<(GbnfElement[], int)> frames) { Frames = frames; }

    internal bool IsComplete => Frames.Count == 0 ||
        Frames.All(f => f.pos >= f.elements.Length);

    internal GrammarStack Clone()
    {
        return new GrammarStack(Frames.Select(f => (f.elements, f.pos)).ToList());
    }

    internal string GetKey()
    {
        return string.Join("|", Frames.Select(f => $"{f.elements.GetHashCode()}:{f.pos}"));
    }
}

/// <summary>Simple GBNF parser producing element arrays per rule.</summary>
internal sealed class GbnfGrammar
{
    private readonly Dictionary<string, GbnfElement[]> _rules = new();

    internal GbnfGrammar(string grammarText) => Parse(grammarText);

    internal List<GrammarStack> GetInitialStacks(string rootRule)
    {
        if (!_rules.TryGetValue(rootRule, out var elements))
            return new();
        var stack = new GrammarStack();
        stack.Frames.Add((elements, 0));
        return ExpandAlternations([stack]);
    }

    /// <summary>
    /// Try to advance a grammar stack by consuming the given text.
    /// Returns all valid successor stacks (may be multiple due to alternations).
    /// </summary>
    internal List<GrammarStack> AdvanceStack(GrammarStack stack, string text)
    {
        var current = new List<GrammarStack> { stack.Clone() };

        foreach (char c in text)
        {
            var next = new List<GrammarStack>();
            foreach (var s in current)
            {
                var advanced = AdvanceByChar(s, c);
                next.AddRange(advanced);
            }
            if (next.Count == 0) return new(); // Dead end
            current = next;
        }
        return current;
    }

    private List<GrammarStack> AdvanceByChar(GrammarStack stack, char c)
    {
        // Expand to get current expected elements
        var expanded = ExpandAlternations([stack]);
        var results = new List<GrammarStack>();

        foreach (var s in expanded)
        {
            var elem = GetCurrentElement(s);
            if (elem == null)
            {
                // Stack complete — can't accept more chars unless it's optional
                continue;
            }

            if (elem is GbnfLiteralElement lit)
            {
                if (lit.CharIndex < lit.Text.Length && lit.Text[lit.CharIndex] == c)
                {
                    var ns = s.Clone();
                    if (lit.CharIndex + 1 >= lit.Text.Length)
                        AdvancePosition(ns); // Literal fully matched
                    else
                        SetCurrentLiteralPos(ns, lit.CharIndex + 1);
                    results.Add(ns);
                }
            }
            else if (elem is GbnfCharClassElement cc)
            {
                if (cc.Matches(c))
                {
                    var ns = s.Clone();
                    AdvancePosition(ns);
                    results.Add(ns);
                }
            }
            else if (elem is GbnfRuleRefElement rref)
            {
                // Push the referenced rule onto the stack
                if (_rules.TryGetValue(rref.Name, out var ruleElements))
                {
                    var ns = s.Clone();
                    ns.Frames.Add((ruleElements, 0));
                    var sub = AdvanceByChar(ns, c);
                    results.AddRange(sub);
                }
            }
        }

        return results;
    }

    private static GbnfElement? GetCurrentElement(GrammarStack stack)
    {
        for (int i = stack.Frames.Count - 1; i >= 0; i--)
        {
            var (elements, pos) = stack.Frames[i];
            if (pos < elements.Length) return elements[pos];
        }
        return null;
    }

    private static void AdvancePosition(GrammarStack stack)
    {
        // Move to next element in innermost frame; pop completed frames
        for (int i = stack.Frames.Count - 1; i >= 0; i--)
        {
            var (elements, pos) = stack.Frames[i];
            stack.Frames[i] = (elements, pos + 1);
            if (pos + 1 < elements.Length) return; // More elements in this frame
            stack.Frames.RemoveAt(i); // Frame complete, pop
        }
    }

    private static void SetCurrentLiteralPos(GrammarStack stack, int charIdx)
    {
        // Update the literal's char position in the current element
        for (int i = stack.Frames.Count - 1; i >= 0; i--)
        {
            var (elements, pos) = stack.Frames[i];
            if (pos < elements.Length && elements[pos] is GbnfLiteralElement lit)
            {
                // Create a new literal with advanced position
                var newElements = (GbnfElement[])elements.Clone();
                newElements[pos] = new GbnfLiteralElement(lit.Text, charIdx);
                stack.Frames[i] = (newElements, pos);
                return;
            }
        }
    }

    private List<GrammarStack> ExpandAlternations(List<GrammarStack> stacks)
    {
        // Expand any alternation elements at the current position
        var result = new List<GrammarStack>();
        foreach (var s in stacks)
        {
            var elem = GetCurrentElement(s);
            if (elem is GbnfAlternationElement alt)
            {
                foreach (var branch in alt.Branches)
                {
                    var ns = s.Clone();
                    // Replace current frame's element with the branch
                    var frame = ns.Frames[^1];
                    var newElements = new GbnfElement[frame.elements.Length - 1 + branch.Length];
                    Array.Copy(frame.elements, 0, newElements, 0, frame.pos);
                    Array.Copy(branch, 0, newElements, frame.pos, branch.Length);
                    Array.Copy(frame.elements, frame.pos + 1, newElements, frame.pos + branch.Length,
                        frame.elements.Length - frame.pos - 1);
                    ns.Frames[^1] = (newElements, frame.pos);
                    result.AddRange(ExpandAlternations([ns]));
                }
            }
            else
            {
                result.Add(s);
            }
        }
        return result;
    }

    // ── Parser ──────────────────────────────────────────────────────────────

    private void Parse(string text)
    {
        var lines = text.Split('\n');
        string? currentName = null;
        string currentExpr = "";

        foreach (var rawLine in lines)
        {
            var line = rawLine.Trim();
            // Strip inline comments (not inside quotes)
            int commentIdx = FindComment(line);
            if (commentIdx >= 0) line = line[..commentIdx].TrimEnd();
            if (line.Length == 0) continue;

            int defIdx = line.IndexOf("::=", StringComparison.Ordinal);
            if (defIdx >= 0)
            {
                if (currentName != null)
                    _rules[currentName] = ParseElements(currentExpr.Trim());
                currentName = line[..defIdx].Trim();
                currentExpr = line[(defIdx + 3)..];
            }
            else if (currentName != null)
                currentExpr += " " + line;
        }
        if (currentName != null)
            _rules[currentName] = ParseElements(currentExpr.Trim());
    }

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
        // Check for top-level alternation
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

            if (expr[i] == '"')
            {
                elem = ParseLiteral(expr, ref i);
            }
            else if (expr[i] == '[')
            {
                elem = ParseCharClass(expr, ref i);
            }
            else if (expr[i] == '(')
            {
                int depth = 1, start = ++i;
                while (i < expr.Length && depth > 0) { if (expr[i]=='(') depth++; else if (expr[i]==')') depth--; i++; }
                var inner = ParseElements(expr[start..(i-1)]);
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

            // Repetition
            if (i < expr.Length && expr[i] == '*') { elem = new GbnfRepeatElement(elem, 0); i++; }
            else if (i < expr.Length && expr[i] == '+') { elem = new GbnfRepeatElement(elem, 1); i++; }
            else if (i < expr.Length && expr[i] == '?') { elem = new GbnfOptionalElement(elem); i++; }
            else if (i < expr.Length && expr[i] == '{')
            {
                i++; int s = i; while (i < expr.Length && expr[i] != '}') i++;
                var parts = expr[s..i].Split(',');
                // For simplicity treat {n} and {n,m} as repeat
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
                sb.Append(expr[i] switch { 'n'=>'\n', 't'=>'\t', 'r'=>'\r', '"'=>'"', '\\'=>'\\', _=>expr[i] });
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
            char from = expr[i]; i++;
            if (i + 1 < expr.Length && expr[i] == '-' && expr[i+1] != ']') { i++; ranges.Add((from, expr[i])); i++; }
            else ranges.Add((from, from));
        }
        if (i < expr.Length) i++;
        return new GbnfCharClassElement(ranges.ToArray(), neg);
    }

    private static List<string> SplitTopLevel(string expr, char sep)
    {
        var result = new List<string>();
        int depth = 0; bool inQ = false; int start = 0;
        for (int i = 0; i < expr.Length; i++)
        {
            if (expr[i] == '"' && (i == 0 || expr[i-1] != '\\')) inQ = !inQ;
            if (!inQ) { if (expr[i]=='('||expr[i]=='[') depth++; else if (expr[i]==')'||expr[i]==']') depth--; }
            if (!inQ && depth == 0 && expr[i] == sep) { result.Add(expr[start..i]); start = i + 1; }
        }
        result.Add(expr[start..]);
        return result;
    }
}

// ── Element types ──
internal abstract class GbnfElement { }
internal sealed class GbnfLiteralElement(string text, int charIndex = 0) : GbnfElement
{
    public string Text => text;
    public int CharIndex => charIndex;
    public bool Matches(char c) => charIndex < text.Length && text[charIndex] == c;
}
internal sealed class GbnfCharClassElement((char from, char to)[] ranges, bool negated) : GbnfElement
{
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
