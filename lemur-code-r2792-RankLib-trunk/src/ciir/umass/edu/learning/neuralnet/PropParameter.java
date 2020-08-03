/*===============================================================================
 * Copyright (c) 2010-2012 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set 
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package ciir.umass.edu.learning.neuralnet;

public class PropParameter {
	//RankNet
	public int current = -1;//index of current data point in the ranked list
	public int[][] pairMap = null;
	public PropParameter(int current, int[][] pairMap)
	{
		this.current = current;
		this.pairMap = pairMap;
	}
	//LambdaRank: RankNet + the following
	public float[][] pairWeight = null;
	public float[][] targetValue = null;
	public PropParameter(int current, int[][] pairMap, float[][] pairWeight, float[][] targetValue)
	{
		this.current = current;
		this.pairMap = pairMap;
		this.pairWeight = pairWeight;
		this.targetValue = targetValue;
	}
	//ListNet
	public float[] labels = null;//relevance label
	public PropParameter(float[] labels)
	{
		this.labels = labels;
	}
}
