package ciir.umass.edu.learning;

import ciir.umass.edu.utilities.RankLibError;

public class DenseDataPoint extends DataPoint {

	public DenseDataPoint(String text) {
		super(text);
	}
	
	public DenseDataPoint(DenseDataPoint dp)
	{
		label = dp.label;
		id = dp.id;
		description = dp.description;
		cached = dp.cached;
		fVals = new float[dp.fVals.length];
		System.arraycopy(dp.fVals, 0, fVals, 0, dp.fVals.length);
	}
	
	@Override
	public float getFeatureValue(int fid)
	{
		if(fid <= 0 || fid >= fVals.length)
		{
			if (missingZero) return 0f;
			throw RankLibError.create("Error in DenseDataPoint::getFeatureValue(): requesting unspecified feature, fid=" + fid);
		}
		if(isUnknown(fVals[fid]))//value for unspecified feature is 0
			return 0;
		return fVals[fid];
	}
	
	@Override
	public void setFeatureValue(int fid, float fval)
	{
		if(fid <= 0 || fid >= fVals.length)
		{
			throw RankLibError.create("Error in DenseDataPoint::setFeatureValue(): feature (id=" + fid + ") not found.");
		}
		fVals[fid] = fval;
	}

	@Override
	public void setFeatureVector(float[] dfVals) {
		//fVals = new float[dfVals.length];
		//System.arraycopy(dfVals, 0, fVals, 0, dfVals.length);
		fVals = dfVals;
	}

	@Override
	public float[] getFeatureVector() {
		return fVals;
	}
}
