/*===============================================================================
 * Copyright (c) 2010-2012 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set 
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package ciir.umass.edu.learning;

import ciir.umass.edu.metric.MetricScorer;
import ciir.umass.edu.utilities.KeyValuePair;
import ciir.umass.edu.utilities.RankLibError;
import ciir.umass.edu.utilities.SimpleMath;

import java.io.BufferedReader;
import java.io.StringReader;
import java.util.Arrays;
import java.util.List;

public class LinearRegRank extends Ranker {

	public static double lambda = 1E-10;//L2-norm regularization parameter
	
	//Local variables
	protected double[] weight = null; 
	
	public LinearRegRank()
	{		
	}
	public LinearRegRank(List<RankList> samples, int[] features, MetricScorer scorer)
	{
		super(samples, features, scorer);
	}
	public void init()
	{
		PRINTLN("Initializing... [Done]");
	}
	public void learn()
	{
		PRINTLN("--------------------------------");
		PRINTLN("Training starts...");
		PRINTLN("--------------------------------");
		PRINT("Learning the least square model... ");
		
		//closed form solution: beta = ((xTx - lambda*I)^(-1)) * (xTy)
		//where x is an n-by-f matrix (n=#data-points, f=#features), y is an n-element vector of relevance labels
		/*int nSample = 0;
		for(int i=0;i<samples.size();i++)
			nSample += samples.get(i).size();*/
		int nVar = DataPoint.getFeatureCount();
		
		double[][] xTx = new double[nVar][];
		for(int i=0;i<nVar;i++)
		{
			xTx[i] = new double[nVar];
			Arrays.fill(xTx[i], 0.0);
		}
		double[] xTy = new double[nVar];
		Arrays.fill(xTy, 0.0);
		
		for(int s=0;s<samples.size();s++)
		{
			RankList rl = samples.get(s);
			for(int i=0;i<rl.size();i++)
			{
				xTy[nVar-1] += rl.get(i).getLabel();
				for(int j=0;j<nVar-1;j++)
				{
					xTy[j] += rl.get(i).getFeatureValue(j+1) * rl.get(i).getLabel();
					for(int k=0;k<nVar;k++)
					{
						double t = (k < nVar-1) ? rl.get(i).getFeatureValue(k+1) : 1f;
						xTx[j][k] += rl.get(i).getFeatureValue(j+1) * t;
					}
				}
				for(int k=0;k<nVar-1;k++)
					xTx[nVar-1][k] += rl.get(i).getFeatureValue(k+1);
				xTx[nVar-1][nVar-1] += 1f;
			}
		}
		if(lambda != 0.0)//regularized
		{
			for(int i=0;i<xTx.length;i++)
				xTx[i][i] += lambda;
		}
		weight = solve(xTx, xTy);
		PRINTLN("[Done]");
		
		scoreOnTrainingData = SimpleMath.round(scorer.score(rank(samples)), 4);
		PRINTLN("---------------------------------");
		PRINTLN("Finished sucessfully.");
		PRINTLN(scorer.name() + " on training data: " + scoreOnTrainingData);

		if(validationSamples != null)
		{
			bestScoreOnValidationData = scorer.score(rank(validationSamples));
			PRINTLN(scorer.name() + " on validation data: " + SimpleMath.round(bestScoreOnValidationData, 4));
		}
		PRINTLN("---------------------------------");
	}
	public double eval(DataPoint p)
	{
		double score = weight[weight.length-1];
		for(int i=0;i<features.length;i++)
			score += weight[i] * p.getFeatureValue(features[i]);
		return score;
	}
	public Ranker createNew()
	{
		return new LinearRegRank();
	}
	public String toString()
	{
		String output = "0:" + weight[0] + " ";		
		for(int i=0;i<features.length;i++)
			output += features[i] + ":" + weight[i] + ((i==weight.length-1)?"":" ");
		return output;
	}
	public String model()
	{
		String output = "## " + name() + "\n";
		output += "## Lambda = " + lambda + "\n";
		output += toString();
		return output;
	}
  @Override
	public void loadFromString(String fullText)
	{
		try {
			String content = "";
			BufferedReader in = new BufferedReader(new StringReader(fullText));

			KeyValuePair kvp = null;
			while((content = in.readLine()) != null)
			{
				content = content.trim();
				if(content.length() == 0)
					continue;
				if(content.indexOf("##")==0)
					continue;
				kvp = new KeyValuePair(content);
				break;
			}
			in.close();

      assert(kvp != null);
			List<String> keys = kvp.keys();
			List<String> values = kvp.values();
			weight = new double[keys.size()];
			features = new int[keys.size()-1];//weight = <weight for each feature, constant>
			int idx = 0;
			for(int i=0;i<keys.size();i++)
			{
				int fid = Integer.parseInt(keys.get(i));
				if(fid > 0)
				{
					features[idx] = fid;
					weight[idx] = Double.parseDouble(values.get(i));
					idx++;
				}
				else
					weight[weight.length-1] = Double.parseDouble(values.get(i));
			}
		}
		catch(Exception ex)
		{
			throw RankLibError.create("Error in LinearRegRank::load(): ", ex);
		}
	}
	public void printParameters()
	{
		PRINTLN("L2-norm regularization: lambda = " + lambda);
	}
	public String name()
	{
		return "Linear Regression";
	}
	/**
	 * Solve a system of linear equations Ax=B, in which A has to be a square matrix with the same length as B
	 * @param A
	 * @param B
	 * @return x
	 */
	protected double[] solve(double[][] A, double[] B)
	{
		if(A.length == 0 || B.length == 0)
		{
			System.out.println("Error: some of the input arrays is empty.");
			System.exit(1);
		}
		if(A[0].length == 0)
		{
			System.out.println("Error: some of the input arrays is empty.");
			System.exit(1);
		}
		if(A.length != B.length)
		{
			System.out.println("Error: Solving Ax=B: A and B have different dimension.");
			System.exit(1);
		}
		
		//init
		double[][] a = new double[A.length][];
		double[] b = new double[B.length];
		System.arraycopy(B, 0, b, 0, B.length);
		for(int i=0;i<a.length;i++)
		{
			a[i] = new double[A[i].length];
			if(i > 0)
			{
				if(a[i].length != a[i-1].length)
				{
					System.out.println("Error: Solving Ax=B: A is NOT a square matrix.");
					System.exit(1);
				}
			}
			System.arraycopy(A[i], 0, a[i], 0, A[i].length);
		}
		//apply the gaussian elimination process to convert the matrix A to upper triangular form
		double pivot = 0.0;
		double multiplier = 0.0;
		for(int j=0;j<b.length-1;j++)//loop through all columns of the matrix A
		{
			pivot = a[j][j];
			for(int i=j+1;i<b.length;i++)//loop through all remaining rows
			{
				multiplier = a[i][j] / pivot;
				//i-th row = i-th row - (multiplier * j-th row) 
				for(int k=j+1;k<b.length;k++)//loop through all remaining elements of the current row, starting at (j+1)
					a[i][k] -= a[j][k] * multiplier;
				b[i] -= b[j] * multiplier;
			}
		}		
		//a*x=b
		//a is now an upper triangular matrix, now the solution x can be obtained with elementary linear algebra
		double[] x = new double[b.length];
		int n = b.length;
		x[n-1] = b[n-1] / a[n-1][n-1];
		for(int i=n-2;i>=0;i--)//walk back up to the first row -- we only need to care about the right to the diagonal
		{
			double val = b[i];
			for(int j=i+1;j<n;j++)
				val -= a[i][j] * x[j];
			x[i] = val / a[i][i];
		}
		
		return x;
	}
}
