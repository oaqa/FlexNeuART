/*===============================================================================
 * Copyright (c) 2010-2012 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set 
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package ciir.umass.edu.learning;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class Sampler {
	protected List<RankList> samples = null;//bag data
	protected List<RankList> remains = null;//out-of-bag data
	public List<RankList> doSampling(List<RankList> samplingPool, float samplingRate, boolean withReplacement)
	{
		Random r = new Random();
		samples = new ArrayList<RankList>();
		int size = (int)(samplingRate * samplingPool.size());
		if(withReplacement)
		{
			int[] used = new int[samplingPool.size()];
			Arrays.fill(used, 0);
			for(int i=0;i<size;i++)
			{
				int selected = r.nextInt(samplingPool.size());
				samples.add(samplingPool.get(selected));			
				used[selected] = 1;
			}
			remains = new ArrayList<RankList>();
			for(int i=0;i<samplingPool.size();i++)
				if(used[i] == 0)
					remains.add(samplingPool.get(i));
		}
		else
		{
			List<Integer> l = new ArrayList<Integer>();
			for(int i=0;i<samplingPool.size();i++)
				l.add(i);
			for(int i=0;i<size;i++)
			{
				int selected = r.nextInt(l.size());
				samples.add(samplingPool.get(l.get(selected)));
				l.remove(selected);
			}
			remains = new ArrayList<RankList>();
			for(int i=0;i<l.size();i++)
				remains.add(samplingPool.get(l.get(i)));
		}
		return samples;
	}
	public List<RankList> getSamples()
	{
		return samples;
	}
	public List<RankList> getRemains()
	{
		return remains;
	}
}
