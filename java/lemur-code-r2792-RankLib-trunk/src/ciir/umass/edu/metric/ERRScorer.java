/*===============================================================================
 * Copyright (c) 2010-2012 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set 
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package ciir.umass.edu.metric;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import ciir.umass.edu.learning.RankList;

/**
 * 
 * @author Van Dang
 * Expected Reciprocal Rank
 */
public class ERRScorer extends MetricScorer {
	
	public static double MAX = 16;//by default, we assume the relevance scale of {0, 1, 2, 3, 4} => g_max = 4 => 2^g_max = 16
	
	public ERRScorer()
	{
		this.k = 10;
	}
	public ERRScorer(int k)
	{
		this.k = k;
	}
	public ERRScorer copy()
	{
		return new ERRScorer();
	}
	/**
	 * Compute ERR at k. NDCG(k) = DCG(k) / DCG_{perfect}(k). Note that the "perfect ranking" must be computed based on the whole list,
	 * not just top-k portion of the list.
	 */
	public double score(RankList rl)
	{
		int size = k;
		if(k > rl.size() || k <= 0)
			size = rl.size();
		
		List<Integer> rel = new ArrayList<Integer>();
		for(int i=0;i<rl.size();i++)
			rel.add((int)rl.get(i).getLabel());
		
		double s = 0.0;
		double p = 1.0;
		for(int i=1;i<=size;i++)
		{
			double R = R(rel.get(i-1)); 
			s += p*R/i;
			p *= (1.0 - R);
		}
		return s;
	}
	public String name()
	{
		return "ERR@" + k;
	}
	private double R(int rel)
	{
		return (double)((1<<rel)-1) / MAX;// (2^rel - 1)/MAX;
	}
	public double[][] swapChange(RankList rl)
	{
		int size = (rl.size() > k) ? k : rl.size();
		int[] labels = new int[rl.size()];
		double[] R = new double[rl.size()];
		double[] np = new double[rl.size()];//p[i] = (1 - p[0])(1 - p[1])...(1-p[i-1])
		double p = 1.0;
		//for(int i=0;i<rl.size();i++)//ignore K, compute changes from the entire ranked list
		for(int i=0;i<size;i++)
		{
			labels[i] = (int)rl.get(i).getLabel();
			R[i] = R(labels[i]);
			np[i] = p * (1.0 - R[i]);
			p *= np[i];
		}
		
		double[][] changes = new double[rl.size()][];
		for(int i=0;i<rl.size();i++)
		{
			changes[i] = new double[rl.size()];
			Arrays.fill(changes[i], 0);
		}
		//for(int i=0;i<rl.size()-1;i++)//ignore K, compute changes from the entire ranked list
		for(int i=0;i<size;i++)
		{
			double v1 = 1.0/(i+1) * (i==0?1:np[i-1]);
			double change = 0;
			for(int j=i+1;j<rl.size();j++)
			{
				if(labels[i] == labels[j])
					change = 0;
				else
				{
					change = v1 * (R[j] - R[i]);
					p = (i==0?1:np[i-1]) * (R[i] - R[j]);
					for(int k=i+1;k<j;k++)
					{
						change += p * R[k]/(1+k);
						p *= 1.0 - R[k];
					}
					change += (np[j-1]*(1.0 - R[j]) * R[i]/(1.0 - R[i]) - np[j-1] * R[j]) / (j+1);
				}
				changes[j][i] = changes[i][j] = change;
			}
		}
		return changes;
	}
}
