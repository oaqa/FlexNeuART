/*===============================================================================
 * Copyright (c) 2010-2012 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set 
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package ciir.umass.edu.metric;

import java.util.Arrays;

import ciir.umass.edu.learning.RankList;

/**
 * @author vdang
 */
public class BestAtKScorer extends MetricScorer {
	
	public BestAtKScorer()
	{
		this.k = 10;
	}
	public BestAtKScorer(int k)
	{
		this.k = k;
	}
	public double score(RankList rl)
	{
		return rl.get(maxToK(rl, k-1)).getLabel();
	}
	public MetricScorer copy()
	{
		return new BestAtKScorer();
	}
	
	/**
	 * Return the position of the best object (e.g. docs with highest degree of relevance) among objects in the range [0..k]
	 * NOTE: If you want best-at-k (i.e. best among top-k), you need maxToK(rl, k-1)
	 * @param l The rank list.
	 * @param k The last position of the range.
	 * @return The index of the best object in the specified range.
	 */
	public int maxToK(RankList rl, int k)
	{
		int size = k;
		if(size < 0 || size > rl.size()-1)
			size = rl.size()-1;
		
		double max = -1.0;
		int max_i = 0;
		for(int i=0;i<=size;i++)
		{
			if(max < rl.get(i).getLabel())
			{
				max = rl.get(i).getLabel();
				max_i = i;
			}
		}
		return max_i;
	}
	public String name()
	{
		return "Best@"+k;
	}
	public double[][] swapChange(RankList rl)
	{
		//FIXME: not sure if this implementation is correct!
		int[] labels = new int[rl.size()];
		int[] best = new int[rl.size()];
		int max = -1;
		int maxVal = -1;		
		int secondMaxVal = -1;//within top-K
		int maxCount = 0;//within top-K
		for(int i=0;i<rl.size();i++)
		{
			int v = (int)rl.get(i).getLabel();
			labels[i] = v;
			if(maxVal < v)
			{
				if(i < k)
				{
					secondMaxVal = maxVal;
					maxCount = 0;
				}
				maxVal = v;
				max = i;
			}
			else if(maxVal == v && i < k)
				maxCount++;
			best[i] = max;
		}
		if(secondMaxVal == -1)
			secondMaxVal = 0;
		
		double[][] changes = new double[rl.size()][];
		for(int i=0;i<rl.size();i++)
		{
			changes[i] = new double[rl.size()];
			Arrays.fill(changes[i], 0);
		}
		//FIXME: THIS IS VERY *INEFFICIENT*
		for(int i=0;i<rl.size()-1;i++)
		{
			for(int j=i+1;j<rl.size();j++)
			{
				double change = 0;
				if(j < k || i >= k)
					change = 0;
				else if(labels[i] == labels[j] || labels[j] == labels[best[k-1]])
					change = 0;
				else if(labels[j] > labels[best[k-1]])
					change = labels[j] - labels[best[i]];
				else if(labels[i] < labels[best[k-1]] || maxCount > 1)
					change = 0;
				else
					change = maxVal - Math.max(secondMaxVal, labels[j]);
				changes[i][j] = changes[j][i] = change;
			}			
		}
		return changes;
	}
}
