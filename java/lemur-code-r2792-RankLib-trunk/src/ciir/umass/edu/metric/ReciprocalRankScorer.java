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
public class ReciprocalRankScorer extends MetricScorer {
	
	public ReciprocalRankScorer()
	{
		this.k = 0;//consider the whole list
	}
	public double score(RankList rl)
	{
		int size = (rl.size() > k) ? k : rl.size();
		int firstRank = -1;
		for(int i=0;i<size && (firstRank==-1);i++)
		{
			if(rl.get(i).getLabel() > 0.0)//relevant
				firstRank = i+1;
		}
		return (firstRank==-1)?0:(1.0f/firstRank);
	}
	public MetricScorer copy()
	{
		return new ReciprocalRankScorer();
	}
	public String name()
	{
		return "RR@"+k;
	}
	public double[][] swapChange(RankList rl)
	{
		int firstRank = -1;
		int secondRank = -1;
		int size = (rl.size() > k) ? k : rl.size();
		for(int i=0;i<size;i++)
		{
			if(rl.get(i).getLabel() > 0.0)//relevant
			{
				if(firstRank==-1)
					firstRank = i;
				else if(secondRank == -1)
					secondRank = i;
			}
		}
		
		//compute the change in RR by swapping each pair
		double[][] changes = new double[rl.size()][];
		for(int i=0;i<rl.size();i++)
		{
			changes[i] = new double[rl.size()];
			Arrays.fill(changes[i], 0);
		}
		
		double rr = 0.0;
		//consider swapping the first rank doc with everything else further down the list
		if(firstRank != -1)
		{
			rr = 1.0 / (firstRank+1);
			for(int j=firstRank+1;j<size;j++)
			{
				if(((int)(rl.get(j).getLabel())) == 0)//non-relevant
				{
					if(secondRank==-1 || j < secondRank)//after the swap, j is now the position of our used-to-be firstRank relevant doc
						changes[firstRank][j] = changes[j][firstRank] = 1.0 / (j+1) - rr;
					else
						changes[firstRank][j] = changes[j][firstRank] = 1.0 / (secondRank+1) - rr;
				}
			}
			for(int j=size;j<rl.size();j++)
				if(((int)(rl.get(j).getLabel())) == 0)//non-relevant
				{
					if(secondRank == -1)
						changes[firstRank][j] = changes[j][firstRank] = - rr;
					else
						changes[firstRank][j] = changes[j][firstRank] = 1.0 / (secondRank+1) - rr;
				}
		}
		else
			firstRank = size;
		
		//now it's time to consider swapping docs at earlier ranks than the first rank with those below it (and *it* too) 
		for(int i=0;i<firstRank;i++)
		{
			for(int j=firstRank;j<rl.size();j++)
			{
				if(rl.get(j).getLabel() > 0)
					changes[i][j] = changes[j][i] = 1.0/(i+1) - rr;
			}
		}			
		return changes;
	}
}
