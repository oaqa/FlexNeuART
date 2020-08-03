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
import ciir.umass.edu.learning.RankList;
import ciir.umass.edu.learning.Ranker;
import ciir.umass.edu.metric.MetricScorer;
import ciir.umass.edu.utilities.KeyValuePair;
import ciir.umass.edu.utilities.RankLibError;
import ciir.umass.edu.utilities.SimpleMath;

import java.io.BufferedReader;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * @author vdang
 * 
 * This class implements the AdaRank algorithm. Here's the paper:
 *  J. Xu and H. Li. AdaRank: a boosting algorithm for information retrieval. In Proc. of SIGIR, pages 391-398, 2007.
 */
public class AdaRank extends Ranker {
	
	//Paramters
	public static int nIteration = 500;
	public static double tolerance = 0.002;
	public static boolean trainWithEnqueue = true;
	public static int maxSelCount = 5;//the max. number of times a feature can be selected consecutively before being removed
	
	protected HashMap<Integer, Integer> usedFeatures = new HashMap<Integer, Integer>();
	protected double[] sweight = null;//sample weight
	protected List<WeakRanker> rankers = null;//alpha
	protected List<Double> rweight = null;//weak rankers' weight
	//to store the best model on validation data (if specified)
	protected List<WeakRanker> bestModelRankers = null;
	protected List<Double> bestModelWeights = null;
	
	//For the implementation of tricks
	int lastFeature = -1;
	int lastFeatureConsecutiveCount = 0;
	boolean performanceChanged = false;
	List<Integer> featureQueue = null;
	protected double[] backupSampleWeight = null;
	protected double backupTrainScore = 0.0;
	protected double lastTrainedScore = -1.0;
	
	public AdaRank()
	{
		
	}
	public AdaRank(List<RankList> samples, int[] features, MetricScorer scorer)
	{
		super(samples, features, scorer);
	}
	
	private void updateBestModelOnValidation()
	{
		bestModelRankers.clear();
		bestModelRankers.addAll(rankers);
		bestModelWeights.clear();
		bestModelWeights.addAll(rweight);
	}
	private WeakRanker learnWeakRanker()
	{
		double bestScore = -1.0;
		WeakRanker bestWR = null;
		for (int i : features) {
			if (featureQueue.contains(i))
				continue;

			if (usedFeatures.get(i) != null)
				continue;

			WeakRanker wr = new WeakRanker(i);
			double s = 0.0;
			for (int j = 0; j < samples.size(); j++) {
				double t = scorer.score(wr.rank(samples.get(j))) * sweight[j];
				s += t;
			}

			if (bestScore < s) {
				bestScore = s;
				bestWR = wr;
			}
		}
		return bestWR;
	}
	private int learn(int startIteration, boolean withEnqueue)
	{
		int t = startIteration;
		for(; t<=nIteration; t++)
		{
			PRINT(new int[]{7}, new String[]{t+""});
			
			WeakRanker bestWR = learnWeakRanker();
			if(bestWR == null)
				break;
			
			if(withEnqueue)
			{
				if(bestWR.getFID() == lastFeature)//this feature is selected twice in a row
				{
					//enqueue this feature
					featureQueue.add(lastFeature);
					//roll back the previous weak ranker since it is based on this "too strong" feature
					rankers.remove(rankers.size()-1);
					rweight.remove(rweight.size()-1);
					copy(backupSampleWeight, sweight);
					bestScoreOnValidationData = 0.0;//no best model just yet
					lastTrainedScore = backupTrainScore;
					PRINTLN(new int[]{8, 9, 9, 9}, new String[]{bestWR.getFID()+"", "", "", "ROLLBACK"});
					continue;
				}
				else
				{
					lastFeature = bestWR.getFID();
					//save the distribution of samples' weight in case we need to rollback
					copy(sweight, backupSampleWeight);
					backupTrainScore = lastTrainedScore;
				}
			}
			
			double num = 0.0;
			double denom = 0.0;
			for(int i=0;i<samples.size();i++)
			{
				double tmp = scorer.score(bestWR.rank(samples.get(i)));
				num += sweight[i]*(1.0 + tmp);
				denom += sweight[i]*(1.0 - tmp);
			}
			
			rankers.add(bestWR);
			double alpha_t = (double) (0.5 * SimpleMath.ln(num/denom));
			rweight.add(alpha_t);
			
			double trainedScore = 0.0;
			//update the distribution of sample weight
			double total = 0.0;
			for (RankList sample : samples) {
				double tmp = scorer.score(rank(sample));
				total += Math.exp(-alpha_t * tmp);
				trainedScore += tmp;
			}
			trainedScore /= samples.size();
			double delta = trainedScore + tolerance - lastTrainedScore;
			String status = (delta>0)?"OK":"DAMN";
			
			if(!withEnqueue)
			{
				if(trainedScore != lastTrainedScore)
				{
					performanceChanged = true;
					lastFeatureConsecutiveCount = 0;
					//all removed features are added back to the pool
					usedFeatures.clear();
				}
				else
				{
					performanceChanged = false;
					if(lastFeature == bestWR.getFID())
					{
						lastFeatureConsecutiveCount++;
						if(lastFeatureConsecutiveCount == maxSelCount)
						{
							status = "F. REM.";
							lastFeatureConsecutiveCount = 0;
							usedFeatures.put(lastFeature, 1);//removed this feature from the pool
						}
					}
					else
					{
						lastFeatureConsecutiveCount = 0;
						//all removed features are added back to the pool
						usedFeatures.clear();
					}
				}
				lastFeature = bestWR.getFID();
			}
			
			PRINT(new int[]{8, 9, }, new String[]{bestWR.getFID()+"", SimpleMath.round(trainedScore, 4)+""});
			if(t % 1==0 && validationSamples != null)
			{
				double scoreOnValidation = scorer.score(rank(validationSamples));
				if(scoreOnValidation > bestScoreOnValidationData)
				{
					bestScoreOnValidationData = scoreOnValidation;
					updateBestModelOnValidation();
				}
				PRINT(new int[]{9, 9}, new String[]{SimpleMath.round(scoreOnValidation, 4)+"", status});
			}
			else
				PRINT(new int[]{9, 9}, new String[]{"", status});
			PRINTLN("");
			
			if(delta <= 0)//stop criteria met
			{
				rankers.remove(rankers.size()-1);
				rweight.remove(rweight.size()-1);
				break;
			}
			
			lastTrainedScore = trainedScore;
			for(int i=0;i<sweight.length;i++)
				sweight[i] *= Math.exp(-alpha_t*scorer.score(rank(samples.get(i))))/total;
		}
		return t;
	}
	
	public void init()
	{
		PRINT("Initializing... ");
		//initialization
		usedFeatures.clear();
		//assign equal weight to all samples
		sweight = new double[samples.size()];
		for(int i=0;i<sweight.length;i++)
			sweight[i] = 1.0f/samples.size();
		backupSampleWeight = new double[sweight.length];
		copy(sweight, backupSampleWeight);
		lastTrainedScore = -1.0;
		
		rankers = new ArrayList<WeakRanker>();
		rweight = new ArrayList<Double>();
		
		featureQueue = new ArrayList<Integer>();
		
		bestScoreOnValidationData = 0.0;
		bestModelRankers = new ArrayList<WeakRanker>();
		bestModelWeights = new ArrayList<Double>();
		
		PRINTLN("[Done]");
	}
	public void learn()
	{		
		PRINTLN("---------------------------");
		PRINTLN("Training starts...");
		PRINTLN("--------------------------------------------------------");
		PRINTLN(new int[]{7, 8, 9, 9, 9}, new String[]{"#iter", "Sel. F.", scorer.name()+"-T", scorer.name()+"-V", "Status"});
		PRINTLN("--------------------------------------------------------");
		
		if(trainWithEnqueue)
		{
			int t = learn(1, true);
			//take care of the enqueued features
			for(int i=featureQueue.size()-1;i>=0;i--)
			{
				featureQueue.remove(i);
				t = learn(t, false);
			}
		}
		else
			learn(1, false);
		
		//if validation data is specified ==> best model on this data has been saved
		//we now restore the current model to that best model
		if(validationSamples != null && bestModelRankers.size()>0)
		{
			rankers.clear();
			rweight.clear();
			rankers.addAll(bestModelRankers);
			rweight.addAll(bestModelWeights);
		}
		
		//print learning score
		scoreOnTrainingData = SimpleMath.round(scorer.score(rank(samples)), 4);
		PRINTLN("--------------------------------------------------------");
		PRINTLN("Finished sucessfully.");
		PRINTLN(scorer.name() + " on training data: " + scoreOnTrainingData);
		if(validationSamples != null)
		{
			bestScoreOnValidationData = scorer.score(rank(validationSamples));
			PRINTLN(scorer.name() + " on validation data: " + SimpleMath.round(bestScoreOnValidationData, 4));
		}
		PRINTLN("---------------------------------");
	}
	public double eval(DataPoint p)
	{
		double score = 0.0;
		for(int j=0;j<rankers.size();j++)
			score += rweight.get(j) * p.getFeatureValue(rankers.get(j).getFID());
		return score;
	}
	public Ranker createNew()
	{
		return new AdaRank();
	}
	public String toString()
	{
		String output = "";
		for(int i=0;i<rankers.size();i++)
			output += rankers.get(i).getFID() + ":" + rweight.get(i) + ((i==rankers.size()-1)?"":" ");
		return output;
	}
	public String model()
	{
		String output = "## " + name() + "\n";
		output += "## Iteration = " + nIteration + "\n";
		output += "## Train with enqueue: " + ((trainWithEnqueue)?"Yes":"No") + "\n";
		output += "## Tolerance = " + tolerance + "\n";
		output += "## Max consecutive selection count = " + maxSelCount + "\n";
		output += toString();
		return output;
	}
	public void loadFromString(String fullText)
	{
		try (BufferedReader in = new BufferedReader(new StringReader(fullText))) {
			String content = "";

			KeyValuePair kvp = null;
			while((content = in.readLine()) != null)
			{
				content = content.trim();
				if(content.length() == 0)
					continue;
				if(content.indexOf("##")==0)
					continue;
				kvp = new KeyValuePair(content);
				break;
			}

			assert(kvp != null);
			
			List<String> keys = kvp.keys();
			List<String> values = kvp.values();
			rweight = new ArrayList<>();
			rankers = new ArrayList<>();
			features = new int[keys.size()];
			for(int i=0;i<keys.size();i++)
			{
				features[i] = Integer.parseInt(keys.get(i));
				rankers.add(new WeakRanker(features[i]));
				rweight.add(Double.parseDouble(values.get(i)));
			}
		}
		catch(Exception ex)
		{
			throw RankLibError.create("Error in AdaRank::load(): ", ex);
		}
	}
	public void printParameters()
	{
		PRINTLN("No. of rounds: " + nIteration);
		PRINTLN("Train with 'enequeue': " + ((trainWithEnqueue)?"Yes":"No"));
		PRINTLN("Tolerance: " + tolerance);
		PRINTLN("Max Sel. Count: " + maxSelCount);
	}
	public String name()
	{
		return "AdaRank";
	}
}
