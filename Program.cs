using System;

using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;


using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace V512Benchmarks
{
  public class MatmulBenchmarks
  {
    [Params(128,256,512)]
    public int Size;

    private float[] A;
    private float[] B;
    private float[] C;

    [GlobalSetup]
    public void GlobalSetup()
    {
      A = new float[Size * Size];
      B = new float[Size * Size];
      C = new float[Size * Size];

      for (int i = 0; i < Size; i++)
      {
        for (int j = 0; j < Size; j++) 
        {
          A[i*Size+j] = 2.0f;
          B[i*Size+j] = 3.0f;
          C[i*Size+j] = 0.0f;
        }
      }
    }

    public unsafe static void NoVectorMatmul(float* a, float* b, float *c, int length)
    {
      for (int i = 0; i < length; i++)
      {
				for (int k = 0; k < length; k++)
				{
					for (int j = 0; j < length; j++)
					{
						c[i*length+j] += a[i*length+k] * b[k*length+j];
					}
				}
      }
    }

    [Benchmark]
    public void RunNoVectorMatmul()
    {
      unsafe
      {
        fixed (float* pA = A, pB = B, pC = C)
        {
          NoVectorMatmul(pA, pB, pC, Size);
        }
      }
    }

    public unsafe static void Vector128Matmul(float* a, float* b, float *c, int length)
    {
      for (int i = 0; i < length; i++)
      {
				for (int k = 0; k < length; k++)
				{
					Vector128<float> sv = Vector128.Create(a[i*length+k]);
					for (int j = 0; j < length; j += Vector128<float>.Count)
					{
						Vector128<float> cv = Sse2.LoadVector128(c+(i*length+j));
						Vector128<float> bv = Sse2.LoadVector128(b+(i*length+j));
						Vector128<float> tv = Sse2.Multiply(bv, sv);
						cv = Sse.Add(tv, cv);
						Sse2.Store(c+(i*length+j), cv);
					}
				}
      }
    }

    [Benchmark]
    public void RunVector128Matmul()
    {
      unsafe
      {
        fixed (float* pA = A, pB = B, pC = C)
        {
          Vector128Matmul(pA, pB, pC, Size);
        }
      }
    }

    public unsafe static void Vector256Matmul(float* a, float* b, float *c, int length)
    {
      for (int i = 0; i < length; i++)
      {
				for (int k = 0; k < length; k++)
				{
					Vector256<float> sv = Vector256.Create(a[i*length+k]);
					for (int j = 0; j < length; j += Vector256<float>.Count)
					{
						Vector256<float> cv = Avx.LoadVector256(c+(i*length+j));
						Vector256<float> bv = Avx.LoadVector256(b+(i*length+j));
						Vector256<float> tv = Avx.Multiply(bv, sv);
						cv = Avx.Add(tv, cv);
						Avx.Store(c+(i*length+j), cv);
					}
				}
      }
    }

    [Benchmark]
    public void RunVector256Matmul()
    {
      unsafe
      {
        fixed (float* pA = A, pB = B, pC = C)
        {
          Vector256Matmul(pA, pB, pC, Size);
        }
      }
    }

  }

  public class Program
  {
    public static void Main(string[] args)
    {
      var summary = BenchmarkRunner.Run<MatmulBenchmarks>();
    }
  }
}
