/*===============================================================================
 * Copyright (c) 2010-2012 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set 
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package ciir.umass.edu.features;

import java.util.HashSet;
import java.util.List;

import ciir.umass.edu.learning.RankList;

/**
 * @author vdang
 *
 * Abstract class for feature normalization
 */
public class Normalizer {
	public void normalize(RankList rl)
	{	
		//need overriding in subclass
	}
	public void normalize(List<RankList> samples)
	{
		for(int i=0;i<samples.size();i++)
			normalize(samples.get(i));
	}
	public void normalize(RankList rl, int[] fids)
	{
		//need overriding in subclass
	}
	public void normalize(List<RankList> samples, int[] fids)
	{
		for(int i=0;i<samples.size();i++)
			normalize(samples.get(i), fids);
	}
	public int[] removeDuplicateFeatures(int[] fids)
	{
		HashSet<Integer> uniqueSet = new HashSet<Integer>();
		for(int i=0;i<fids.length;i++)
			if(!uniqueSet.contains(fids[i]))
				uniqueSet.add(fids[i]);
		fids = new int[uniqueSet.size()];
		int fi=0;
		for(Integer i : uniqueSet)
			fids[fi++] = i.intValue();
		return fids;
	}
	public String name()
	{
		//need overriding in subclass
		return "";
	}
}
