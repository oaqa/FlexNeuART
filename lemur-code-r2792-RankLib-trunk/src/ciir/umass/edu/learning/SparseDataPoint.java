/*===============================================================================
 * Copyright (c) 2010-2012 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set 
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package ciir.umass.edu.learning;

import ciir.umass.edu.utilities.RankLibError;

import java.util.Arrays;

/**
 * Implements a sparse data point using a compressed sparse row data structure
 * @author Siddhartha Bagaria
 */
public class SparseDataPoint extends DataPoint {

	// Access pattern of the feature values
	private enum accessPattern {SEQUENTIAL, RANDOM};
	private static accessPattern searchPattern = accessPattern.RANDOM;
	
	// Profiling variables
	// private static int numCalls = 0;
	// private static float avgTime = 0;
	
	// The feature ids for known values
	int fIds[];
	
	// The feature values for corresponding Ids
	//float fVals[]; //moved to the parent class
	
	// Internal search optimizers. Currently unused.
	int lastMinId = -1;
	int lastMinPos = -1;
	
	public SparseDataPoint(String text) {
		super(text);
	}

	public SparseDataPoint(SparseDataPoint dp)
 	{
		label = dp.label;
		id = dp.id;
		description = dp.description;
		cached = dp.cached;
		fIds = new int[dp.fIds.length];
		fVals = new float[dp.fVals.length];
		System.arraycopy(dp.fIds, 0, fIds, 0, dp.fIds.length);
		System.arraycopy(dp.fVals, 0, fVals, 0, dp.fVals.length);
 	}
 	
	private int locate(int fid) {
		if (searchPattern == accessPattern.SEQUENTIAL)
		{
			if (lastMinId > fid)
			{
				lastMinId = -1;
				lastMinPos = -1;
			}
			while (lastMinPos < knownFeatures && lastMinId < fid)
				lastMinId = fIds[++lastMinPos];
			if (lastMinId == fid)
				return lastMinPos;
		}
		else if (searchPattern == accessPattern.RANDOM)
		{
			int pos = Arrays.binarySearch(fIds, fid);
			if (pos >= 0)
				return pos;
		}
		else
			System.err.println("Invalid search pattern specified for sparse data points.");

		return -1;
	}

	public boolean hasFeature(int fid) {
		return locate(fid) != -1;
	}

	@Override
	public float getFeatureValue(int fid)
	{
		//long time = System.nanoTime();
		if(fid <= 0 || fid > getFeatureCount())
		{
			if (missingZero) return 0f;
			throw RankLibError.create("Error in SparseDataPoint::getFeatureValue(): requesting unspecified feature, fid=" + fid);
		}
		int pos = locate(fid);
		//long completedIn = System.nanoTime() - time;
		//avgTime = (avgTime*numCalls + completedIn)/(++numCalls);
		//System.out.println("getFeatureValue average time: "+avgTime);
		if(pos >= 0)
			return fVals[pos];
		
		return 0; // Should ideally be returning unknown?
	}
	
	@Override
	public void setFeatureValue(int fid, float fval) 
	{
		if(fid <= 0 || fid > getFeatureCount())
		{
			throw RankLibError.create("Error in SparseDataPoint::setFeatureValue(): feature (id=" + fid + ") out of range.");
		}
		int pos = locate(fid);
		if(pos >= 0)
			fVals[pos] = fval;
		else
		{
			System.err.println("Error in SparseDataPoint::setFeatureValue(): feature (id=" + fid + ") not found.");
			System.exit(1);
		}
	}
	
	@Override
	public void setFeatureVector(float[] dfVals)
	{
		fIds = new int[knownFeatures];
		fVals = new float[knownFeatures];
		int pos = 0;
		for (int i=1; i<dfVals.length; i++)
		{
			if (!isUnknown(dfVals[i]))
			{
				fIds[pos] = i;
				fVals[pos] = dfVals[i];
				pos++;
			}
		}
		assert(pos == knownFeatures);
	}	
	
	@Override
	public float[] getFeatureVector()
	{
		float[] dfVals = new float[fIds[knownFeatures -1]];
		Arrays.fill(dfVals, UNKNOWN);
		for (int i=0; i<knownFeatures; i++)
			dfVals[fIds[i]] = fVals[i];
		return dfVals;
	}
}