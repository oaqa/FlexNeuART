/*===============================================================================
 * Copyright (c) 2010-2012 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set 
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package ciir.umass.edu.learning.boosting;

import ciir.umass.edu.learning.DataPoint;

/**
 * @author vdang
 * 
 * Weak rankers for RankBoost.
 */
public class RBWeakRanker {
	private int  fid = -1;
	private double threshold = 0.0;
	
	public RBWeakRanker(int  fid, double threshold)
	{
		this.fid = fid;
		this.threshold = threshold;
	}
	public int score(DataPoint p)
	{
		if(p.getFeatureValue(fid) > threshold)
			return 1;
		return 0;
	}
	public int getFid()
	{
		return fid;
	}
	public double getThreshold()
	{
		return threshold;
	}
	public String toString()
	{
		return fid + ":" + threshold;
	}
}
