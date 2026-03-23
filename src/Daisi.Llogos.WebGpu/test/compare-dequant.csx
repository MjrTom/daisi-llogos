// Quick C# script to print dequanted Q4_0 values using llogos's GGUF reader
// Run with: dotnet script test/compare-dequant.csx
#r "../Daisi.Llogos/bin/Debug/net10.0/Daisi.Llogos.dll"
using Daisi.Llogos.Gguf;
using var stream = File.OpenRead(@"C:\GGUFS\tinyllama-1.1b-chat-v1.0.Q4_0.gguf");
var gguf = GgufFile.Read(stream);
Console.WriteLine($"Tensors: {gguf.Tensors.Count}, TensorDataOffset: {gguf.TensorDataOffset}");

// Find embedding
var emb = gguf.Tensors.First(t => t.Name == "token_embd.weight");
Console.WriteLine($"token_embd: type={emb.Type} offset={emb.Offset} dims=[{string.Join(",", emb.Dimensions)}]");

// Read first block of embedding and dequant
var embData = gguf.ReadTensorData(stream, emb);
Console.WriteLine($"Raw bytes[0..17]: {BitConverter.ToString(embData, 0, 18).Replace("-", " ")}");

// Dequant first 32 elements
var scale = BitConverter.ToHalf(embData, 0);
Console.Write($"Scale (f16): {scale} = {(float)scale}\n");
for (int j = 0; j < 8; j++) {
    int lo = (embData[2 + j] & 0x0F) - 8;
    int hi = ((embData[2 + j] >> 4) & 0x0F) - 8;
    Console.Write($"  [{j}]={((float)scale * lo):F6}  [{j+16}]={((float)scale * hi):F6}\n");
}

// Also check blk.0.attn_q first block
var q0 = gguf.Tensors.First(t => t.Name == "blk.0.attn_q.weight");
Console.WriteLine($"\nblk.0.attn_q: type={q0.Type} offset={q0.Offset}");
var q0Data = gguf.ReadTensorData(stream, q0);
var q0Scale = BitConverter.ToHalf(q0Data, 0);
Console.Write($"Q scale: {(float)q0Scale}\n");
for (int j = 0; j < 4; j++) {
    int lo = (q0Data[2 + j] & 0x0F) - 8;
    Console.Write($"  q[{j}]={((float)q0Scale * lo):F6}\n");
}
