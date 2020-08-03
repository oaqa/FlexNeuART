/*===============================================================================
 * Copyright (c) 2010-2012 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set 
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package ciir.umass.edu.features;

import java.util.Arrays;

import ciir.umass.edu.learning.DataPoint;
import ciir.umass.edu.learning.RankList;

/**
 * @author vdang
 */
public class SumNormalizor extends Normalizer {
	@Override
	public void normalize(RankList rl) {
		if(rl.size() == 0)
		{
			System.out.println("Error in SumNormalizor::normalize(): The input ranked list is empty");
			System.exit(1);
		}
		int nFeature = DataPoint.getFeatureCount();
		double[] norm = new double[nFeature];
		Arrays.fill(norm, 0);
		for(int i=0;i<rl.size();i++)
		{
			DataPoint dp = rl.get(i);
			for(int j=1;j<=nFeature;j++)
				norm[j-1] += Math.abs(dp.getFeatureValue(j));
		}
		for(int i=0;i<rl.size();i++)
		{
			DataPoint dp = rl.get(i);
			for(int j=1;j<=nFeature;j++)
			{
				if(norm[j-1] > 0)
					dp.setFeatureValue(j, (float)(dp.getFeatureValue(j)/norm[j-1]));
			}
		}
	}
	@Override
	public void normalize(RankList rl, int[] fids) {
		if(rl.size() == 0)
		{
			System.out.println("Error in SumNormalizor::normalize(): The input ranked list is empty");
			System.exit(1);
		}
		
		//remove duplicate features from the input @fids ==> avoid normalizing the same features multiple times
		fids = removeDuplicateFeatures(fids);
				
		double[] norm = new double[fids.length];
		Arrays.fill(norm, 0);
		for(int i=0;i<rl.size();i++)
		{
			DataPoint dp = rl.get(i);
			for(int j=0;j<fids.length;j++)
				norm[j] += Math.abs(dp.getFeatureValue(fids[j]));
		}
		for(int i=0;i<rl.size();i++)
		{
			DataPoint dp = rl.get(i);
			for(int j=0;j<fids.length;j++)
				if(norm[j] > 0)
					dp.setFeatureValue(fids[j], (float)(dp.getFeatureValue(fids[j])/norm[j]));
		}
	}
	public String name()
	{
		return "sum";
	}
}
