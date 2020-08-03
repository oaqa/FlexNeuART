package ciir.umass.edu.eval;

import ciir.umass.edu.stats.RandomPermutationTest;
import ciir.umass.edu.utilities.FileUtils;
import ciir.umass.edu.utilities.RankLibError;
import ciir.umass.edu.utilities.SimpleMath;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

public class Analyzer {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub

		String directory = "";
		String baseline = "";
		if(args.length < 2)
		{
			System.out.println("Usage: java -cp bin/RankLib.jar ciir.umass.edu.eval.Analyzer <Params>");
			System.out.println("Params:");
			System.out.println("\t-all <directory>\tDirectory of performance files (one per system)");
			System.out.println("\t-base <file>\t\tPerformance file for the baseline (MUST be in the same directory)");
			System.out.println("\t[ -np ] \t\tNumber of permutation (Fisher randomization test) [default=" + RandomPermutationTest.nPermutation + "]");
			return;
		}
		
		for(int i=0;i<args.length;i++)
		{
			if(args[i].compareTo("-all")==0)
				directory = args[++i];
			else if(args[i].compareTo("-base")==0)
				baseline = args[++i];
			else if(args[i].compareTo("-np")==0)
				RandomPermutationTest.nPermutation = Integer.parseInt(args[++i]);
		}
		
		Analyzer a = new Analyzer();
		a.compare(directory, baseline);
		//a.compare("output/", "ca.feature.base");
	}

	static class Result {
		int status = 0;//success
		int win = 0;
		int loss = 0;
		int[] countByImprovementRange = null;
	}
	
	private RandomPermutationTest randomizedTest = new RandomPermutationTest();
	private static double[] improvementRatioThreshold = new double[]{-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1000};
	private int indexOfZero = 4;
	private int locateSegment(double value)
	{
		if(value > 0)
		{
			for(int i=indexOfZero;i<improvementRatioThreshold.length;i++)
				if(value <= improvementRatioThreshold[i])
					return i;
		}
		else if(value < 0)
		{
			for(int i=0;i<=indexOfZero;i++)
				if(value < improvementRatioThreshold[i])
					return i;
		}
		return -1;
	}
	
	/**
	 * Read performance (in some measure of effectiveness) file. Expecting: id [space]* metric-text [space]* performance
	 * @param filename
	 * @return Mapping from ranklist-id --> performance
	 */
	public HashMap<String, Double> read(String filename)
	{
		HashMap<String, Double> performance = new HashMap<String, Double>();		
		try (BufferedReader in = FileUtils.smartReader(filename))
		{
			String content = "";
			while((content = in.readLine()) != null)
			{
				content = content.trim();
				if(content.length() == 0)
					continue;
				
				//expecting: id [space]* metric-text [space]* performance
				while(content.contains("  "))
					content = content.replace("  ", " ");
				content = content.replace(" ", "\t");
				String[] s = content.split("\t");
				//String measure = s[0];
				String id = s[1];
				double p = Double.parseDouble(s[2]);
				performance.put(id, p);
			}
			in.close();
			System.out.println("Reading " + filename + "... " + performance.size() + " ranked lists [Done]");
		}
		catch(IOException ex)
		{
			throw RankLibError.create(ex);
		}
		return performance;
	}
	/**
	 * Compare the performance of a set of systems to that of a baseline system 
	 * @param directory Contain files denoting the performance of the target systems to be compared  
	 * @param baseFile Performance file for the baseline system
	 */
	public void compare(String directory, String baseFile)
	{
		directory = FileUtils.makePathStandard(directory);
		List<String> targets = FileUtils.getAllFiles2(directory);//ONLY filenames are stored 
		for(int i=0;i<targets.size();i++)
		{
			if(targets.get(i).compareTo(baseFile) == 0)
			{
				targets.remove(i);
				i--;
			}
			else
				targets.set(i, directory+targets.get(i));//convert filename to full path
		}
		compare(targets, directory+baseFile);
	}
	/**
	 * Compare the performance of a set of systems to that of a baseline system 
	 * @param targetFiles Performance files of the target systems to be compared (full path)
	 * @param baseFile Performance file for the baseline system
	 */
	public void compare(List<String> targetFiles, String baseFile)
	{
		HashMap<String, Double> base = read(baseFile);
		List<HashMap<String, Double>> targets = new ArrayList<HashMap<String, Double>>();
		for(int i=0;i<targetFiles.size();i++)
		{
			HashMap<String, Double> hm = read(targetFiles.get(i));
			targets.add(hm);
		}
		Result[] rs = compare(base, targets);
		
		//overall comparison
		System.out.println("");
		System.out.println("");
		System.out.println("Overall comparison");
		System.out.println("------------------------------------------------------------------------");
		System.out.println("System\tPerformance\tImprovement\tWin\tLoss\tp-value");
		System.out.println(FileUtils.getFileName(baseFile) + " [baseline]\t" + SimpleMath.round(base.get("all").doubleValue(), 4));
		for(int i=0;i<rs.length;i++)
		{
			if(rs[i].status == 0)
			{
				double delta = targets.get(i).get("all") - base.get("all");
				double dp = delta*100/ base.get("all");
				String msg = FileUtils.getFileName(targetFiles.get(i)) + "\t" + SimpleMath.round(targets.get(i).get("all").doubleValue(), 4);
				msg += "\t" + ((delta>0)?"+":"") + SimpleMath.round(delta, 4) + " (" + ((delta>0)?"+":"") + SimpleMath.round(dp, 2) + "%)";
				msg += "\t" + rs[i].win + "\t" + rs[i].loss;
				msg += "\t" + randomizedTest.test(targets.get(i), base) + "";
				System.out.println(msg);
			}
			else
				System.out.println("WARNING: [" + targetFiles.get(i) + "] skipped: NOT comparable to the baseline due to different ranked list IDs.");
		}
		//in more details
		System.out.println("");
		System.out.println("");
		System.out.println("Detailed break down");
		System.out.println("------------------------------------------------------------------------");
		String header = "";
		String[] tmp = new String[improvementRatioThreshold.length];
		for(int i=0;i<improvementRatioThreshold.length;i++)
		{
			String t = (int)(improvementRatioThreshold[i]*100) + "%";
			if(improvementRatioThreshold[i] > 0)
				t = "+" + t;
			tmp[i] = t;
		}
		header += "[ < " + tmp[0] + ")\t";
		for(int i=0;i<improvementRatioThreshold.length-2;i++)
		{
			if(i >= indexOfZero)
				header += "(" + tmp[i] + ", " + tmp[i+1] + "]\t";
			else
				header += "[" + tmp[i] + ", " + tmp[i+1] + ")\t";
		}
		header += "( > " + tmp[improvementRatioThreshold.length-2] + "]";
		System.out.println("\t" + header);
		
		for(int i=0;i<targets.size();i++)
		{
			String msg = FileUtils.getFileName(targetFiles.get(i));
			for(int j=0;j<rs[i].countByImprovementRange.length;j++)
				msg += "\t" + rs[i].countByImprovementRange[j];
			System.out.println(msg);
		}
	}
	/**
	 * Compare the performance of a set of systems to that of a baseline system
	 * @param base
	 * @param targets
	 * @return
	 */
	public Result[] compare(HashMap<String, Double> base, List<HashMap<String, Double>> targets)
	{
		//comparative statistics
		Result[] rs = new Result[targets.size()];
		for(int i=0;i<targets.size();i++)
			rs[i] = compare(base, targets.get(i));			
		return rs;
	}
	/**
	 * Compare the performance of a target system to that of a baseline system
	 * @param base
	 * @param target
	 * @return
	 */
	public Result compare(HashMap<String, Double> base, HashMap<String, Double> target)
	{
		Result r = new Result();
		if(base.size() != target.size())
		{
			r.status = -1;
			return r;
		}
		
		r.countByImprovementRange = new int[improvementRatioThreshold.length];
		Arrays.fill(r.countByImprovementRange, 0);
		for(String key: base.keySet())
		{
			if(!target.containsKey(key))
			{
				r.status = -2;
				return r;
			}
			if(key.compareTo("all") == 0)
				continue;
			double p = base.get(key);
			double pt = target.get(key);
			if(pt > p)
				r.win++;
			else if(pt < p)
				r.loss++;
			double change = pt - p;
			if(change != 0)//only interested in cases where <change != 0>
				r.countByImprovementRange[locateSegment(change)]++;
		}
		return r;
	}
}
