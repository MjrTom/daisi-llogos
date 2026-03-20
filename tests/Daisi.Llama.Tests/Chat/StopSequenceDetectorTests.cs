using Daisi.Llama.Chat;

namespace Daisi.Llama.Tests.Chat;

public class StopSequenceDetectorTests
{
    [Fact]
    public void DetectsCompleteStopSequence()
    {
        var detector = new StopSequenceDetector(["<|im_end|>"]);

        Assert.False(detector.Process("Hello", out var text1));
        Assert.Equal("Hello", text1);

        Assert.True(detector.Process("<|im_end|>", out var text2));
        Assert.Equal("", text2);
    }

    [Fact]
    public void DetectsStopSequenceSplitAcrossTokens()
    {
        var detector = new StopSequenceDetector(["<|im_end|>"]);

        Assert.False(detector.Process("Hello", out var text1));
        Assert.Equal("Hello", text1);

        // Partial match — "<|im" could be start of stop sequence
        Assert.False(detector.Process("<|im", out var text2));
        Assert.Equal("", text2); // buffered

        // Complete the stop sequence
        Assert.True(detector.Process("_end|>", out var text3));
        Assert.Equal("", text3);
    }

    [Fact]
    public void EmitsBufferedTextWhenPartialMatchFails()
    {
        var detector = new StopSequenceDetector(["<|im_end|>"]);

        Assert.False(detector.Process("Hello", out var text1));
        Assert.Equal("Hello", text1);

        // Partial match
        Assert.False(detector.Process("<|im", out var text2));
        Assert.Equal("", text2);

        // This doesn't continue the stop sequence, so buffer flushes
        Assert.False(detector.Process("possible", out var text3));
        Assert.Equal("<|impossible", text3);
    }

    [Fact]
    public void HandlesMultipleStopSequences()
    {
        var detector = new StopSequenceDetector(["<|im_end|>", "</tool_call>"]);

        Assert.False(detector.Process("text", out _));

        Assert.True(detector.Process("</tool_call>", out var text));
        Assert.Equal("", text);
    }

    [Fact]
    public void EmitsTextBeforeStopSequence()
    {
        var detector = new StopSequenceDetector(["STOP"]);

        Assert.False(detector.Process("Hello ", out var t1));
        Assert.Equal("Hello ", t1);

        Assert.True(detector.Process("worldSTOP", out var t2));
        Assert.Equal("world", t2);
    }

    [Fact]
    public void FlushReturnsBufferedContent()
    {
        var detector = new StopSequenceDetector(["<|im_end|>"]);

        Assert.False(detector.Process("text<|im", out _));

        var flushed = detector.Flush();
        Assert.Equal("<|im", flushed);
    }

    [Fact]
    public void ResetClearsState()
    {
        var detector = new StopSequenceDetector(["<|im_end|>"]);

        detector.Process("text<|im", out _);
        detector.Reset();

        var flushed = detector.Flush();
        Assert.Equal("", flushed);
    }

    [Fact]
    public void NoStopSequences_PassesThrough()
    {
        var detector = new StopSequenceDetector([]);

        Assert.False(detector.Process("Hello", out var text1));
        Assert.Equal("Hello", text1);

        Assert.False(detector.Process(" world", out var text2));
        Assert.Equal(" world", text2);
    }
}
