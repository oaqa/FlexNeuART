package ciir.umass.edu.stats;

public class BasicStats {
	public static double mean(double[] values)
	{
		double mean = 0.0;
		if(values.length == 0)
		{
			System.out.println("Error in BasicStats::mean(): Empty input array.");
			System.exit(1);
		}
		for(int i=0;i<values.length;i++)
			mean += values[i];
		return mean/values.length;
	}
}
