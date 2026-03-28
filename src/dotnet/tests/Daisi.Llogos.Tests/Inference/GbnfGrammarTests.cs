using Daisi.Llogos.Inference;

namespace Daisi.Llogos.Tests.Inference;

/// <summary>
/// Tests for the GBNF grammar engine's parsing and constraint logic.
/// </summary>
public class GbnfGrammarTests
{
    private static readonly string JsonGrammar =
        "root   ::= object\r\nvalue  ::= object | array | string | number | (\"true\" | \"false\" | \"null\") ws\r\n\r\nobject ::=\r\n  \"{\" ws (\r\n            string \":\" ws value\r\n    (\",\" ws string \":\" ws value)*\r\n  )? \"}\" ws\r\n\r\narray  ::=\r\n  \"[\" ws (\r\n            value\r\n    (\",\" ws value)*\r\n  )? \"]\" ws\r\n\r\nstring ::=\r\n  \"\\\"\" (\r\n    [^\"\\\\\\x7F\\x00-\\x1F] |\r\n    \"\\\\\" ([\"\\\\bfnrt] | \"u\" [0-9a-fA-F]{4}) # escapes\r\n  )* \"\\\"\" ws\r\n\r\nnumber ::= (\"-\"? ([0-9] | [1-9] [0-9]{0,15})) (\".\" [0-9]+)? ([eE] [-+]? [0-9] [1-9]{0,15})? ws\r\n\r\n# Optional space: by convention, applied in this grammar after literal chars when allowed\r\nws ::= | \" \" | \"\\n\" [ \\t]{0,20}";

    [Fact]
    public void JsonGrammar_Accepts_EmptyObject()
    {
        var grammar = new GbnfGrammar(JsonGrammar);
        var stacks = grammar.GetInitialStacks("root");
        Assert.NotEmpty(stacks);

        // Should accept "{}"
        stacks = AdvanceAll(grammar, stacks, "{}");
        Assert.NotEmpty(stacks);
    }

    [Fact]
    public void JsonGrammar_Accepts_SimpleObject()
    {
        var grammar = new GbnfGrammar(JsonGrammar);
        var stacks = grammar.GetInitialStacks("root");

        stacks = AdvanceAll(grammar, stacks, "{\"key\":\"val\"}");
        Assert.NotEmpty(stacks);
    }

    [Fact]
    public void JsonGrammar_Accepts_ObjectWithSpaces()
    {
        var grammar = new GbnfGrammar(JsonGrammar);
        var stacks = grammar.GetInitialStacks("root");

        stacks = AdvanceAll(grammar, stacks, "{ \"key\": \"val\" }");
        Assert.NotEmpty(stacks);
    }

    [Fact]
    public void JsonGrammar_Rejects_InvalidStart()
    {
        var grammar = new GbnfGrammar(JsonGrammar);
        var stacks = grammar.GetInitialStacks("root");

        // JSON grammar root is object, which must start with {
        stacks = AdvanceAll(grammar, stacks, "hello");
        Assert.Empty(stacks);
    }

    [Fact]
    public void JsonGrammar_Accepts_OpenBrace_ThenQuote()
    {
        var grammar = new GbnfGrammar(JsonGrammar);
        var stacks = grammar.GetInitialStacks("root");

        // Step 1: accept {
        stacks = AdvanceAll(grammar, stacks, "{");
        Assert.NotEmpty(stacks);

        // Step 2: after {, should accept " (start of key string)
        var afterQuote = AdvanceAll(grammar, stacks, "\"");
        Assert.NotEmpty(afterQuote);
    }

    [Fact]
    public void JsonGrammar_Accepts_OpenBrace_ThenCloseBrace()
    {
        var grammar = new GbnfGrammar(JsonGrammar);
        var stacks = grammar.GetInitialStacks("root");

        stacks = AdvanceAll(grammar, stacks, "{");
        Assert.NotEmpty(stacks);

        // After {, should accept } (empty object)
        var afterClose = AdvanceAll(grammar, stacks, "}");
        Assert.NotEmpty(afterClose);
    }

    [Fact]
    public void SimpleGrammar_Accepts_Literal()
    {
        var grammar = new GbnfGrammar("root ::= \"hello\"");
        var stacks = grammar.GetInitialStacks("root");
        Assert.NotEmpty(stacks);

        stacks = AdvanceAll(grammar, stacks, "hello");
        Assert.NotEmpty(stacks);
    }

    [Fact]
    public void SimpleGrammar_Alternation()
    {
        var grammar = new GbnfGrammar("root ::= \"a\" | \"b\"");
        var stacks = grammar.GetInitialStacks("root");

        var afterA = AdvanceAll(grammar, stacks, "a");
        Assert.NotEmpty(afterA);

        var afterB = AdvanceAll(grammar, stacks, "b");
        Assert.NotEmpty(afterB);

        var afterC = AdvanceAll(grammar, stacks, "c");
        Assert.Empty(afterC);
    }

    [Fact]
    public void SimpleGrammar_RuleRef()
    {
        var grammar = new GbnfGrammar("root ::= greeting\ngreeting ::= \"hi\"");
        var stacks = grammar.GetInitialStacks("root");

        stacks = AdvanceAll(grammar, stacks, "hi");
        Assert.NotEmpty(stacks);
    }

    [Fact]
    public void SimpleGrammar_Optional()
    {
        var grammar = new GbnfGrammar("root ::= \"a\" \"b\"?  \"c\"");
        var stacks = grammar.GetInitialStacks("root");

        // "ac" should work (skip optional)
        var stacks1 = AdvanceAll(grammar, stacks, "ac");
        Assert.NotEmpty(stacks1);

        // "abc" should also work (take optional)
        var stacks2 = AdvanceAll(grammar, stacks, "abc");
        Assert.NotEmpty(stacks2);
    }

    [Fact]
    public void SimpleGrammar_Repeat()
    {
        var grammar = new GbnfGrammar("root ::= \"a\" [b]* \"c\"");
        var stacks = grammar.GetInitialStacks("root");

        // "ac" - zero repeats
        var s1 = AdvanceAll(grammar, stacks, "ac");
        Assert.NotEmpty(s1);

        // "abc" - one repeat
        var s2 = AdvanceAll(grammar, stacks, "abc");
        Assert.NotEmpty(s2);

        // "abbc" - two repeats
        var s3 = AdvanceAll(grammar, stacks, "abbc");
        Assert.NotEmpty(s3);
    }

    [Fact]
    public void JsonGrammar_StepByStep_SimpleObject()
    {
        var grammar = new GbnfGrammar(JsonGrammar);
        var stacks = grammar.GetInitialStacks("root");

        // Step through {"key":"val"} one character at a time
        var chars = "{\"key\":\"val\"}";
        for (int i = 0; i < chars.Length; i++)
        {
            var c = chars[i].ToString();
            var next = new List<GrammarStack>();
            foreach (var s in stacks)
                next.AddRange(grammar.AcceptChars(s, c));

            Assert.True(next.Count > 0, $"Grammar rejected char '{chars[i]}' (0x{(int)chars[i]:X2}) at position {i}. Consumed so far: \"{chars[..i]}\"");
            stacks = next;
        }
    }

    [Fact]
    public void AlternationInRepeat()
    {
        // This tests the pattern from JSON string rule: ( charclass | escape )*
        var grammar = new GbnfGrammar("root ::= \"\\\"\" ([a-z] | [A-Z])* \"\\\"\"");
        var stacks = grammar.GetInitialStacks("root");
        var chars = "\"aB\"";
        for (int i = 0; i < chars.Length; i++)
        {
            var next = new List<GrammarStack>();
            foreach (var s in stacks)
                next.AddRange(grammar.AcceptChars(s, chars[i].ToString()));
            Assert.True(next.Count > 0, $"Rejected char '{chars[i]}' at pos {i} after \"{chars[..i]}\"");
            stacks = next;
        }
    }

    [Fact]
    public void JsonStringRule_MultiChar()
    {
        // Exact string rule from JSON grammar
        var grammar = new GbnfGrammar(
            "root ::= \"\\\"\" ( [^\"\\\\\\x7F\\x00-\\x1F] | \"\\\\\" ([\"\\\\bfnrt] | \"u\" [0-9a-fA-F]{4}) )* \"\\\"\"");

        // Dump rules
        var rules = grammar.GetRules();
        var output = new System.Text.StringBuilder();
        for (int r = 0; r < rules.Count; r++)
        {
            var elems = string.Join(", ", rules[r].Elements.Select(e => e.Type.ToString()));
            output.AppendLine($"[{r}] {rules[r].Name} ({rules[r].Elements.Length} elems) = [{elems}]");
        }

        var stacks = grammar.GetInitialStacks("root");
        var chars = "\"key\"";
        for (int i = 0; i < chars.Length; i++)
        {
            var next = new List<GrammarStack>();
            foreach (var s in stacks)
                next.AddRange(grammar.AcceptChars(s, chars[i].ToString()));
            Assert.True(next.Count > 0, $"Rejected '{chars[i]}' at {i} after \"{chars[..i]}\"\nRules:\n{output}");
            stacks = next;
        }
    }

    [Fact]
    public void AlternationInRepeat_WithRuleRef()
    {
        // Like JSON string but with explicit rule refs
        var grammar = new GbnfGrammar("root ::= \"{\" str \"}\"\nstr ::= \"\\\"\" ([a-z] | \"\\\\\" [a-z])* \"\\\"\"");
        var stacks = grammar.GetInitialStacks("root");
        var chars = "{\"abc\"}";
        for (int i = 0; i < chars.Length; i++)
        {
            var next = new List<GrammarStack>();
            foreach (var s in stacks)
                next.AddRange(grammar.AcceptChars(s, chars[i].ToString()));
            Assert.True(next.Count > 0, $"Rejected char '{chars[i]}' at pos {i} after \"{chars[..i]}\"");
            stacks = next;
        }
    }

    [Fact]
    public void StringRule_Accepts_MultiChar()
    {
        // Isolated string rule test
        var grammar = new GbnfGrammar("root ::= \"\\\"\" [a-z]* \"\\\"\"");
        var stacks = grammar.GetInitialStacks("root");

        // Should accept "key"
        var chars = "\"key\"";
        for (int i = 0; i < chars.Length; i++)
        {
            var next = new List<GrammarStack>();
            foreach (var s in stacks)
                next.AddRange(grammar.AcceptChars(s, chars[i].ToString()));
            Assert.True(next.Count > 0, $"Rejected char '{chars[i]}' at pos {i} after \"{chars[..i]}\"");
            stacks = next;
        }
    }

    [Fact]
    public void CharClassRepeat_MultiChar()
    {
        // Even simpler: just a char class repeat
        var grammar = new GbnfGrammar("root ::= [a-z]*");
        var stacks = grammar.GetInitialStacks("root");

        foreach (var c in "abc")
        {
            var next = new List<GrammarStack>();
            foreach (var s in stacks)
                next.AddRange(grammar.AcceptChars(s, c.ToString()));
            Assert.True(next.Count > 0, $"Rejected char '{c}'");
            stacks = next;
        }
    }

    [Fact]
    public void NestedRuleRef_StringInObject()
    {
        // String via rule ref inside an object-like structure
        var grammar = new GbnfGrammar(
            "root ::= \"{\" str \"}\"\n" +
            "str ::= \"\\\"\" [a-z]* \"\\\"\"");
        var stacks = grammar.GetInitialStacks("root");

        var chars = "{\"key\"}";
        for (int i = 0; i < chars.Length; i++)
        {
            var next = new List<GrammarStack>();
            foreach (var s in stacks)
                next.AddRange(grammar.AcceptChars(s, chars[i].ToString()));
            Assert.True(next.Count > 0, $"Rejected char '{chars[i]}' at pos {i} after \"{chars[..i]}\"");
            stacks = next;
        }
    }

    [Fact]
    public void NestedRuleRef_OptionalGroupWithRepeat()
    {
        // Object with optional content containing string repeat
        var grammar = new GbnfGrammar(
            "root ::= \"{\" (str)? \"}\"\n" +
            "str ::= \"\\\"\" [a-z]* \"\\\"\"");
        var stacks = grammar.GetInitialStacks("root");

        var chars = "{\"key\"}";
        for (int i = 0; i < chars.Length; i++)
        {
            var next = new List<GrammarStack>();
            foreach (var s in stacks)
                next.AddRange(grammar.AcceptChars(s, chars[i].ToString()));
            Assert.True(next.Count > 0, $"Rejected char '{chars[i]}' at pos {i} after \"{chars[..i]}\"");
            stacks = next;
        }
    }

    [Fact]
    public void Debug_JsonGrammar_StackTrace()
    {
        var grammar = new GbnfGrammar(JsonGrammar);
        var stacks = grammar.GetInitialStacks("root");

        var chars = "{\"ke";
        var output = new System.Text.StringBuilder();

        // Dump all rules
        var rules = grammar.GetRules();
        output.AppendLine($"=== RULES ({rules.Count}) ===");
        for (int r = 0; r < rules.Count; r++)
        {
            var rule = rules[r];
            var elems = string.Join(", ", rule.Elements.Select(e =>
                e.Type == GrammarElemType.RuleRef ? $"Ref({e.RuleName})" :
                e.Type == GrammarElemType.CharNot ? $"CharNot({string.Join("", e.Ranges!.Select(rng => rng.from == rng.to ? $"{rng.from}" : $"{rng.from}-{rng.to}"))})" :
                e.Ranges?.Length == 1 && e.Ranges[0].from == e.Ranges[0].to ? $"Char('{e.Ranges[0].from}')" :
                $"CharRange({string.Join("", e.Ranges!.Select(rng => rng.from == rng.to ? $"{rng.from}" : $"{rng.from}-{rng.to}"))})"));
            output.AppendLine($"  [{r}] {rule.Name} = [{elems}]");
        }
        output.AppendLine();
        output.AppendLine($"Initial stacks: {stacks.Count}");
        foreach (var s in stacks)
            output.AppendLine($"  [{string.Join(", ", s.Select(f => $"({f.ruleIdx}:{f.elemIdx})"))}]");

        for (int i = 0; i < chars.Length; i++)
        {
            var next = new List<GrammarStack>();
            foreach (var s in stacks)
                next.AddRange(grammar.AcceptChars(s, chars[i].ToString()));

            output.AppendLine($"\nAfter '{chars[i]}' ({next.Count} stacks):");
            foreach (var s in next)
            {
                var desc = string.Join(", ", s.Select(f =>
                {
                    var r = grammar.GetRules();
                    var rule = r[f.ruleIdx];
                    var elemDesc = f.elemIdx < rule.Elements.Length ? rule.Elements[f.elemIdx].Type.ToString() : "END";
                    if (rule.Elements.Length > f.elemIdx && rule.Elements[f.elemIdx].Type == GrammarElemType.RuleRef)
                        elemDesc += $"→{rule.Elements[f.elemIdx].RuleName}";
                    return $"({rule.Name}[{f.ruleIdx}]:{f.elemIdx}={elemDesc})";
                }));
                output.AppendLine($"  [{desc}]");
            }

            if (next.Count == 0)
            {
                output.AppendLine($"REJECTED at position {i}");
                break;
            }
            stacks = next;
        }

        Assert.Fail(output.ToString());
    }

    private static List<GrammarStack> AdvanceAll(GbnfGrammar grammar, List<GrammarStack> stacks, string text)
    {
        var current = stacks;
        foreach (char c in text)
        {
            var next = new List<GrammarStack>();
            foreach (var s in current)
            {
                next.AddRange(grammar.AcceptChars(s, c.ToString()));
            }
            if (next.Count == 0) return [];
            current = next;
        }
        return current;
    }
}
