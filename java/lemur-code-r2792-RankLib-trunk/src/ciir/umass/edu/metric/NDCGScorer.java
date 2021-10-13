/*===============================================================================
 * Copyright (c) 2010-2012 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set 
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package ciir.umass.edu.metric;

import ciir.umass.edu.learning.RankList;
import ciir.umass.edu.utilities.FileUtils;
import ciir.umass.edu.utilities.RankLibError;
import ciir.umass.edu.utilities.Sorter;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

/**
 * @author vdang
 */
public class NDCGScorer extends DCGScorer {
	
	protected HashMap<String, Double>  idealGains = null;
	
	public NDCGScorer()
	{
		super();
		idealGains = new HashMap<>();
	}
	public NDCGScorer(int k)
	{
		super(k);
		idealGains = new HashMap<>();
	}
	public MetricScorer copy()
	{
		return new NDCGScorer();
	}
	public void loadExternalRelevanceJudgment(String qrelFile)
	{
		//Queries with external relevance judgment will have their cached ideal gain value overridden 
		try (BufferedReader in = FileUtils.smartReader(qrelFile))
		{
			String content = "";
			String lastQID = "";
			List<Integer> rel = new ArrayList<Integer>();
			int nQueries = 0;
			while((content = in.readLine()) != null)
			{
				content = content.trim();
				if(content.length() == 0)
					continue;
				String[] s = content.split(" ");
				String qid = s[0].trim();
				//String docid = s[2].trim();
				int label = (int) Math.rint(Double.parseDouble(s[3].trim()));
				if(lastQID.compareTo("")!=0 && lastQID.compareTo(qid)!=0)
				{
					int size = (rel.size() > k) ? k : rel.size();
					int[] r = new int[rel.size()];
					for(int i=0;i<rel.size();i++)
						r[i] = rel.get(i);
					double ideal = getIdealDCG(r, size);
					idealGains.put(lastQID, ideal);
					rel.clear();
					nQueries++;
				}
				lastQID = qid;
				rel.add(label);
			}
			if(rel.size() > 0)
			{
				int size = (rel.size() > k) ? k : rel.size();
				int[] r = new int[rel.size()];
				for(int i=0;i<rel.size();i++)
					r[i] = rel.get(i);
				double ideal = getIdealDCG(r, size);
				idealGains.put(lastQID, ideal);
				rel.clear();
				nQueries++;
			}
			System.out.println("Relevance judgment file loaded. [#q=" + nQueries + "]");
		} catch (IOException ex) {
			throw RankLibError.create("Error in NDCGScorer::loadExternalRelevanceJudgment(): ", ex);
		}
	}
	
	/**
	 * Compute NDCG at k. NDCG(k) = DCG(k) / DCG_{perfect}(k). Note that the "perfect ranking" must be computed based on the whole list,
	 * not just top-k portion of the list.
	 */
	public double score(RankList rl)
	{
		if(rl.size() == 0)
			return 0;

		int size = k;
		if(k > rl.size() || k <= 0)
			size = rl.size();
		
		int[] rel = getRelevanceLabels(rl);
		
		double ideal = 0;
		Double d = idealGains.get(rl.getID());
		if(d != null)
			ideal = d;
		else
		{
			ideal = getIdealDCG(rel, size);
			idealGains.put(rl.getID(), ideal);
		}
		
		if(ideal <= 0.0)//I mean precisely "="
			return 0.0;
		
		return getDCG(rel, size)/ideal;
	}
	public double[][] swapChange(RankList rl)
	{
		int size = (rl.size() > k) ? k : rl.size();
		//compute the ideal ndcg
		int[] rel = getRelevanceLabels(rl);
		double ideal = 0;
		Double d = idealGains.get(rl.getID());
		if(d != null)
			ideal = d;
		else
		{
			ideal = getIdealDCG(rel, size);
			//idealGains.put(rl.getID(), ideal);//DO *NOT* do caching here. It's not thread-safe.
		}
		
		double[][] changes = new double[rl.size()][];
		for(int i=0;i<rl.size();i++)
		{
			changes[i] = new double[rl.size()];
			Arrays.fill(changes[i], 0);
		}
		
		for(int i=0;i<size;i++)
			for(int j=i+1;j<rl.size();j++)
				if(ideal > 0)
					changes[j][i] = changes[i][j] = (discount(i) - discount(j)) * (gain(rel[i]) - gain(rel[j])) / ideal;

		return changes;
	}
	public String name()
	{
		return "NDCG@"+k;
	}
	
	private double getIdealDCG(int[] rel, int topK)
	{
		int[] idx = Sorter.sort(rel, false);
		double dcg = 0;
		for(int i=0;i<topK;i++)
			dcg += gain(rel[idx[i]]) * discount(i);
		return dcg;
	}
}
