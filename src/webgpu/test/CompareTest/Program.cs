using Daisi.Llogos.Gguf;
var gguf = GgufFile.Read(File.OpenRead(@"C:\GGUFS\tinyllama-1.1b-chat-v1.0.Q4_0.gguf"));
using var stream = File.OpenRead(@"C:\GGUFS\tinyllama-1.1b-chat-v1.0.Q4_0.gguf");
int E=2048,nH=32,nKV=4,hD=64,ffn=5632,V=32000,nL=22,mSL=64;
float eps=1e-5f,theta=10000f;
var emb=DQ(gguf,stream,"token_embd.weight");var oN=LF(gguf,stream,"output_norm.weight",E);var oW=DQ(gguf,stream,"output.weight");
var L=new(float[]an,float[]q,float[]k,float[]v,float[]o,float[]fn,float[]g,float[]u,float[]d)[nL];
for(int i=0;i<nL;i++)L[i]=(LF(gguf,stream,$"blk.{i}.attn_norm.weight",E),DQ(gguf,stream,$"blk.{i}.attn_q.weight"),DQ(gguf,stream,$"blk.{i}.attn_k.weight"),DQ(gguf,stream,$"blk.{i}.attn_v.weight"),DQ(gguf,stream,$"blk.{i}.attn_output.weight"),LF(gguf,stream,$"blk.{i}.ffn_norm.weight",E),DQ(gguf,stream,$"blk.{i}.ffn_gate.weight"),DQ(gguf,stream,$"blk.{i}.ffn_up.weight"),DQ(gguf,stream,$"blk.{i}.ffn_down.weight"));
var kvK=new float[nL][];var kvV=new float[nL][];
for(int i=0;i<nL;i++){kvK[i]=new float[nKV*mSL*hD];kvV[i]=new float[nKV*mSL*hD];}
int sl=0;
// Process 2 tokens: BOS(1) then The(1576)
foreach(int tok in new[]{1,1576}){
  var h=new float[E];Array.Copy(emb,tok*E,h,0,E);
  for(int layer=0;layer<nL;layer++){
    var lw=L[layer];var res=(float[])h.Clone();
    var n1=RN(h,lw.an,E,eps);
    var qP=MM(lw.q,n1,nH*hD,E);var kP=MM(lw.k,n1,nKV*hD,E);var vP=MM(lw.v,n1,nKV*hD,E);
    RP(qP,hD,sl,theta);RP(kP,hD,sl,theta);
    for(int kh=0;kh<nKV;kh++)for(int d=0;d<hD;d++){kvK[layer][kh*mSL*hD+sl*hD+d]=kP[kh*hD+d];kvV[layer][kh*mSL*hD+sl*hD+d]=vP[kh*hD+d];}
    int cs=sl+1;float sc=1f/MathF.Sqrt(hD);int hpg=nH/nKV;
    var ao=new float[nH*hD];
    for(int head=0;head<nH;head++){int kvH=head/hpg;var scores=new float[cs];
      for(int pos=0;pos<cs;pos++){float dot=0;for(int d=0;d<hD;d++)dot+=qP[head*hD+d]*kvK[layer][kvH*mSL*hD+pos*hD+d];scores[pos]=dot*sc;}
      float mx=scores.Max();float se=0;for(int i=0;i<cs;i++){scores[i]=MathF.Exp(scores[i]-mx);se+=scores[i];}
      for(int i=0;i<cs;i++)scores[i]/=se;
      for(int d=0;d<hD;d++){float a=0;for(int pos=0;pos<cs;pos++)a+=scores[pos]*kvV[layer][kvH*mSL*hD+pos*hD+d];ao[head*hD+d]=a;}}
    var oP=MM(lw.o,ao,E,nH*hD);for(int i=0;i<E;i++)h[i]=oP[i]+res[i];
    var r2=(float[])h.Clone();var n2=RN(h,lw.fn,E,eps);
    var gO=MM(lw.g,n2,ffn,E);var uO=MM(lw.u,n2,ffn,E);var ff=new float[ffn];
    for(int i=0;i<ffn;i++)ff[i]=(gO[i]/(1+MathF.Exp(-gO[i])))*uO[i];
    var dn=MM(lw.d,ff,E,ffn);for(int i=0;i<E;i++)h[i]=dn[i]+r2[i];
    if(layer==0)Console.WriteLine($"Tok {tok} L0: [{F(h,5)}]");
  }
  sl++;
  var fn2=RN(h,oN,E,eps);var lg=MM(oW,fn2,V,E);
  var top=lg.Select((v,i)=>(v,i)).OrderByDescending(x=>x.v).Take(5);
  Console.WriteLine($"Tok {tok} top-5: {string.Join(", ",top.Select(t=>$"{t.i}({t.v:F2})"))}");
}
static string F(float[]a,int n)=>string.Join(", ",a.Take(n).Select(v=>v.ToString("F6")));
static float[]RN(float[]i,float[]w,int n,float e){double s=0;for(int j=0;j<n;j++)s+=(double)i[j]*i[j];float r=(float)Math.Sqrt(s/n+e);var o=new float[n];for(int j=0;j<n;j++)o[j]=(i[j]/r)*w[j];return o;}
static float[]MM(float[]w,float[]i,int M,int K){var o=new float[M];for(int r=0;r<M;r++){double s=0;int off=r*K;for(int k=0;k<K;k++)s+=(double)w[off+k]*i[k];o[r]=(float)s;}return o;}
static void RP(float[]d,int hD,int pos,float th){for(int pi=0;pi<d.Length/2;pi++){int hp=pi%(hD/2);float df=(hp*2f)/hD,fr=1f/MathF.Pow(th,df),a=pos*fr,c=MathF.Cos(a),s=MathF.Sin(a);int i0=pi*2,i1=i0+1;float x=d[i0],y=d[i1];d[i0]=x*c-y*s;d[i1]=x*s+y*c;}}
static float[]DQ(GgufFile g,Stream s,string n){var t=g.Tensors.First(x=>x.Name==n);var r=g.ReadTensorData(s,t);if(t.Type==GgmlType.F32){var f=new float[t.ElementCount];Buffer.BlockCopy(r,0,f,0,r.Length);return f;}if(t.Type==GgmlType.Q4_0)return DQ4(r,(int)t.ElementCount);if(t.Type==GgmlType.Q6_K)return DQ6(r,(int)t.ElementCount);throw new($"{t.Type}");}
static float[]LF(GgufFile g,Stream s,string n,int sz){var t=g.Tensors.First(x=>x.Name==n);var r=g.ReadTensorData(s,t);var f=new float[sz];Buffer.BlockCopy(r,0,f,0,sz*4);return f;}
static float[]DQ4(byte[]d,int ec){var r=new float[ec];int bc=(ec+31)/32;for(int b=0;b<bc;b++){int bo=b*18;float sc=(float)BitConverter.ToHalf(d,bo);for(int j=0;j<16;j++){byte q=d[bo+2+j];r[b*32+j]=sc*((q&0xF)-8);r[b*32+j+16]=sc*(((q>>4)&0xF)-8);}}return r;}
static float[]DQ6(byte[]d,int ec){var r=new float[ec];int bc=(ec+255)/256;for(int b=0;b<bc;b++){int bo=b*210;float dd=(float)BitConverter.ToHalf(d,bo+208);for(int n=0;n<256;n+=128)for(int l=0;l<32;l++){int is_=n/16;byte ql0=d[bo+n/2+l],ql1=d[bo+n/2+l+32],qh=d[bo+128+n/4+l];int q1=((ql0&0xF)|((qh&3)<<4))-32,q2=((ql1&0xF)|(((qh>>2)&3)<<4))-32,q3=((ql0>>4)|(((qh>>4)&3)<<4))-32,q4=((ql1>>4)|(((qh>>6)&3)<<4))-32;sbyte s0=(sbyte)d[bo+192+is_],s2=(sbyte)d[bo+192+is_+2],s4=(sbyte)d[bo+192+is_+4],s6=(sbyte)d[bo+192+is_+6];int oi=b*256+n+l;if(oi<ec)r[oi]=dd*s0*q1;if(oi+32<ec)r[oi+32]=dd*s2*q2;if(oi+64<ec)r[oi+64]=dd*s4*q3;if(oi+96<ec)r[oi+96]=dd*s6*q4;}}return r;}
