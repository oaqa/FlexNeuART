
/*===============================================================================
 * Copyright (c) 2010-2015 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set 
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package ciir.umass.edu.learning;

import ciir.umass.edu.metric.MetricScorer;
import ciir.umass.edu.utilities.FileUtils;
import ciir.umass.edu.utilities.MergeSorter;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import java.util.Set;

//- Some Java 7 file utilities for creating directories
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.attribute.FileAttribute;
import java.nio.file.attribute.PosixFilePermission;
import java.nio.file.attribute.PosixFilePermissions;



/**
 * @author vdang
 * 
 * This class implements the generic Ranker interface. Each ranking algorithm implemented has to extend this class. 
 */
public abstract class Ranker {
	public static boolean verbose = true;

	protected List<RankList> samples = new ArrayList<RankList>();//training samples
	protected int[] features = null;
	protected MetricScorer scorer = null;
	protected double scoreOnTrainingData = 0.0;
	protected double bestScoreOnValidationData = 0.0;
	
	protected List<RankList> validationSamples = null;
	
	protected Ranker()
	{

	}
	protected Ranker(List<RankList> samples, int[] features, MetricScorer scorer)
	{
		this.samples = samples;
		this.features = features;
		this.scorer = scorer;
	}
	
	//Utility functions
	public void setTrainingSet(List<RankList> samples)
	{
		this.samples = samples;
	
	}
	public void setFeatures(int[] features)
	{
		this.features = features;	
	}
	public void setValidationSet(List<RankList> samples)
	{
		this.validationSamples = samples;
	}
	public void setMetricScorer(MetricScorer scorer)
	{
		this.scorer = scorer;
	}
	
	public double getScoreOnTrainingData()
	{
		return scoreOnTrainingData;
	}
	public double getScoreOnValidationData()
	{
		return bestScoreOnValidationData;
	}

	public int[] getFeatures()
	{
		return features;
	}
	
	public RankList rank(RankList rl)
	{
		double[] scores = new double[rl.size()];
		for(int i=0;i<rl.size();i++)
			scores[i] = eval(rl.get(i));
		int[] idx = MergeSorter.sort(scores, false);
		return new RankList(rl, idx);
	}

	public List<RankList> rank(List<RankList> l)
	{
		List<RankList> ll = new ArrayList<RankList>();
		for(int i=0;i<l.size();i++)
			ll.add(rank(l.get(i)));
		return ll;
	}

        //- Create the model file directory to write models into if not already there
	public void save(String modelFile) 
	{
              // Determine if the directory to write to exists.  If not, create it.
              Path parentPath = Paths.get(modelFile).toAbsolutePath().getParent();
            
              // Create the directory if it doesn't exist. Give it 755 perms
                if (Files.notExists (parentPath)) {
                     try {
                          Set<PosixFilePermission> perms = PosixFilePermissions.fromString ("rwxr-xr-x");
                          FileAttribute<Set<PosixFilePermission>> attr = PosixFilePermissions.asFileAttribute (perms);
                          Path outputDir = Files.createDirectory (parentPath, attr);
                     }
                     catch (Exception e) {
                          System.out.println ("Error creating kcv model file directory " + modelFile);
		     }         
                }
            
		FileUtils.write(modelFile, "ASCII", model());
	}
	
	protected void PRINT(String msg)
	{
		if(verbose)
			System.out.print(msg);
	}

	protected void PRINTLN(String msg)
	{
		if(verbose)
			System.out.println(msg);
	}

	protected void PRINT(int[] len, String[] msgs)
	{
		if(verbose)
		{
			for(int i=0;i<msgs.length;i++)
			{
				String msg = msgs[i];
				if(msg.length() > len[i])
					msg = msg.substring(0, len[i]);
				else
					while(msg.length() < len[i])
						msg += " ";
				System.out.print(msg + " | ");
			}
		}
	}
	protected void PRINTLN(int[] len, String[] msgs)
	{
		PRINT(len, msgs);
		PRINTLN("");
	}
	protected void PRINTTIME()
	{
		DateFormat dateFormat = new SimpleDateFormat("MM/dd HH:mm:ss");
		Date date = new Date();
		System.out.println(dateFormat.format(date));
	}
	protected void PRINT_MEMORY_USAGE()
	{
		System.out.println("***** " + Runtime.getRuntime().freeMemory() + " / " + Runtime.getRuntime().maxMemory());
	}
	
	protected void copy(double[] source, double[] target)
	{
		for(int j=0;j<source.length;j++)
			target[j] = source[j];
	}
	
	/**
	 * HAVE TO BE OVER-RIDDEN IN SUB-CLASSES
	 */
	public abstract void init();
	public abstract void learn();
	public double eval(DataPoint p)
	{
		return -1.0;
	}

  public abstract Ranker createNew();
  public abstract String toString();
  public abstract String model();
  public abstract void loadFromString(String fullText);
  public abstract String name();
  public abstract void printParameters();
}
