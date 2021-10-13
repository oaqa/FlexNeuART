package ciir.umass.edu.stats;

import java.util.HashMap;
import java.util.Random;

/**
 * Randomized permutation test. Adapted from Michael Bendersky's Python script.
 * @author vdang
 *
 */
public class RandomPermutationTest extends SignificanceTest {
	
	public static int nPermutation = 10000;
	private static String[] pad = new String[]{"", "0", "00", "000", "0000", "00000", "000000", "0000000", "00000000", "000000000"};
	
	/**
	 * Run the randomization test
	 * @param baseline
	 * @param target
	 * @return
	 */
	public double test(HashMap<String, Double> target, HashMap<String, Double> baseline)
	{
		double[] b = new double[baseline.keySet().size()];//baseline
		double[] t = new double[target.keySet().size()];//target
		int c = 0;
		for(String key : baseline.keySet())
		{
			b[c] = baseline.get(key).doubleValue();
			t[c] = target.get(key).doubleValue();
			c++;
		}
		double trueDiff = Math.abs(BasicStats.mean(b) - BasicStats.mean(t));
		double pvalue = 0.0;
		double[] pb = new double[baseline.keySet().size()];//permutation of baseline
		double[] pt = new double[target.keySet().size()];//permutation of target
		for(int i=0;i<nPermutation;i++)
		{
			char[] bits = randomBitVector(b.length).toCharArray();
			for(int j=0;j<b.length;j++)
			{
				if(bits[j] == '0')
				{	
					pb[j] = b[j];
					pt[j] = t[j];
				}
				else
				{
					pb[j] = t[j];
					pt[j] = b[j];
				}
			}
			double pDiff = Math.abs(BasicStats.mean(pb) - BasicStats.mean(pt));
			if(pDiff >= trueDiff)
				pvalue += 1.0;
		}
		return pvalue/nPermutation;
	}
	
	/**
	 * Generate a random bit vector of a certain size
	 * @param size
	 * @return
	 */
	private String randomBitVector(int size)
	{
		Random r = new Random();
		String output = "";
		for(int i=0;i<(size/10)+1;i++)
		{
			int x = (int)((1<<10) * r.nextDouble());
			String s = Integer.toBinaryString(x);
			if(s.length() == 11)
				output += s.substring(1);
			else
				output += pad[10-s.length()] + s;
		}
		return output;
	}
}
