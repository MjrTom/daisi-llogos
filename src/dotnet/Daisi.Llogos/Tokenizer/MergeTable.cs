namespace Daisi.Llogos.Tokenizer;

/// <summary>
/// BPE merge rules ordered by priority. Lower rank = higher priority (merge first).
/// Built from the "tokenizer.ggml.merges" GGUF metadata array.
/// Each merge string is "tokenA tokenB" (space-separated pair).
/// </summary>
public sealed class MergeTable
{
    private readonly Dictionary<(string, string), int> _mergeRank;

    public MergeTable(string[] merges)
    {
        _mergeRank = new Dictionary<(string, string), int>(merges.Length);
        for (int i = 0; i < merges.Length; i++)
        {
            var spaceIdx = merges[i].IndexOf(' ');
            if (spaceIdx > 0)
            {
                var a = merges[i][..spaceIdx];
                var b = merges[i][(spaceIdx + 1)..];
                _mergeRank.TryAdd((a, b), i);
            }
        }
    }

    /// <summary>Total number of merge rules.</summary>
    public int Count => _mergeRank.Count;

    /// <summary>
    /// Get the rank (priority) of a merge pair. Lower rank = higher priority.
    /// Returns -1 if the pair is not in the merge table.
    /// </summary>
    public int GetRank(string a, string b) => _mergeRank.GetValueOrDefault((a, b), -1);
}
