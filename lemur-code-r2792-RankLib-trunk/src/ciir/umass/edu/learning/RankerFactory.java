/*===============================================================================
 * Copyright (c) 2010-2012 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set 
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package ciir.umass.edu.learning;

import ciir.umass.edu.learning.boosting.AdaRank;
import ciir.umass.edu.learning.boosting.RankBoost;
import ciir.umass.edu.learning.neuralnet.LambdaRank;
import ciir.umass.edu.learning.neuralnet.ListNet;
import ciir.umass.edu.learning.neuralnet.RankNet;
import ciir.umass.edu.learning.tree.LambdaMART;
import ciir.umass.edu.learning.tree.MART;
import ciir.umass.edu.learning.tree.RFRanker;
import ciir.umass.edu.metric.MetricScorer;
import ciir.umass.edu.utilities.FileUtils;
import ciir.umass.edu.utilities.RankLibError;

import java.io.BufferedReader;
import java.io.StringReader;
import java.util.HashMap;
import java.util.List;

/**
 * @author vdang
 * 
 * This class implements the Ranker factory. All ranking algorithms implemented have to be recognized in this class. 
 */
public class RankerFactory {

	protected Ranker[] rFactory = new Ranker[]{new MART(), new RankBoost(), new RankNet(), new AdaRank(), new CoorAscent(), new LambdaRank(), new LambdaMART(), new ListNet(), new RFRanker(), new LinearRegRank()};
	protected static HashMap<String, RANKER_TYPE> map = new HashMap<String, RANKER_TYPE>();
	
	public RankerFactory()
	{
		map.put(createRanker(RANKER_TYPE.MART).name().toUpperCase(), RANKER_TYPE.MART);
		map.put(createRanker(RANKER_TYPE.RANKNET).name().toUpperCase(), RANKER_TYPE.RANKNET);
		map.put(createRanker(RANKER_TYPE.RANKBOOST).name().toUpperCase(), RANKER_TYPE.RANKBOOST);
		map.put(createRanker(RANKER_TYPE.ADARANK).name().toUpperCase(), RANKER_TYPE.ADARANK);
		map.put(createRanker(RANKER_TYPE.COOR_ASCENT).name().toUpperCase(), RANKER_TYPE.COOR_ASCENT);
		map.put(createRanker(RANKER_TYPE.LAMBDARANK).name().toUpperCase(), RANKER_TYPE.LAMBDARANK);
		map.put(createRanker(RANKER_TYPE.LAMBDAMART).name().toUpperCase(), RANKER_TYPE.LAMBDAMART);
		map.put(createRanker(RANKER_TYPE.LISTNET).name().toUpperCase(), RANKER_TYPE.LISTNET);
		map.put(createRanker(RANKER_TYPE.RANDOM_FOREST).name().toUpperCase(), RANKER_TYPE.RANDOM_FOREST);
		map.put(createRanker(RANKER_TYPE.LINEAR_REGRESSION).name().toUpperCase(), RANKER_TYPE.LINEAR_REGRESSION);
	}	
	public Ranker createRanker(RANKER_TYPE type)
	{
		return rFactory[type.ordinal() - RANKER_TYPE.MART.ordinal()].createNew();
	}
	public Ranker createRanker(RANKER_TYPE type, List<RankList> samples, int[] features, MetricScorer scorer)
	{
		Ranker r = createRanker(type);
		r.setTrainingSet(samples);
		r.setFeatures(features);
		r.setMetricScorer(scorer);
		return r;
	}
	@SuppressWarnings("unchecked")
	public Ranker createRanker(String className)
	{
		Ranker r = null;
		try {
			Class c = Class.forName(className);
			r = (Ranker) c.newInstance();
		}
		catch (ClassNotFoundException e) {
			System.out.println("Could find the class \"" + className + "\" you specified. Make sure the jar library is in your classpath.");
			e.printStackTrace();
			System.exit(1);
		}
		catch (InstantiationException e) {
			System.out.println("Cannot create objects from the class \"" + className + "\" you specified.");
			e.printStackTrace();
			System.exit(1);
		}
		catch (IllegalAccessException e) {
			System.out.println("The class \"" + className + "\" does not implement the Ranker interface.");
			e.printStackTrace();
			System.exit(1);
		}
		return r;
	}
	public Ranker createRanker(String className, List<RankList> samples, int[] features, MetricScorer scorer)
	{
		Ranker r = createRanker(className);
		r.setTrainingSet(samples);
		r.setFeatures(features);
		r.setMetricScorer(scorer);
		return r;
	}
	public Ranker loadRankerFromFile(String modelFile)
	{
    return loadRankerFromString(FileUtils.read(modelFile, "ASCII"));
	}
  public Ranker loadRankerFromString(String fullText)
  {
    try (BufferedReader in = new BufferedReader(new StringReader(fullText))) {
			Ranker r;
      String content = in.readLine();//read the first line to get the name of the ranking algorithm
      content = content.replace("## ", "").trim();
      System.out.println("Model:\t\t" + content);
      r = createRanker(map.get(content.toUpperCase()));
      r.loadFromString(fullText);
			return r;
    }
    catch(Exception ex)
    {
			throw RankLibError.create(ex);
    }
  }
}
