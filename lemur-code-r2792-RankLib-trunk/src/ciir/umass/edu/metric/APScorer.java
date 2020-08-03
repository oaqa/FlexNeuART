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

import java.io.BufferedReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;

/**
 * @author vdang
 * This class implements MAP (Mean Average Precision)
 */
public class APScorer extends MetricScorer {
	//This class computes MAP from the *WHOLE* ranked list. "K" will be completely ignored.
	//The reason is, if you want MAP@10, you really should be using NDCG@10 or ERR@10 instead.
	
	public HashMap<String, Integer> relDocCount = null;
	
	public APScorer()
	{
		this.k = 0;//consider the whole list
	}
	public MetricScorer copy()
	{
		return new APScorer();
	}
	public void loadExternalRelevanceJudgment(String qrelFile)
	{
		relDocCount = new HashMap<>();
		try (BufferedReader in = FileUtils.smartReader(qrelFile)) {
			String content = "";
			while((content = in.readLine()) != null)
			{
				content = content.trim();
				if(content.length() == 0)
					continue;
				String[] s = content.split(" ");
				String qid = s[0].trim();
				//String docid = s[2].trim();
				int label = (int) Math.rint(Double.parseDouble(s[3].trim()));
				if(label > 0) {
					int prev = relDocCount.getOrDefault(qid, 0);
					relDocCount.put(qid, prev+1);
				}
			}

			System.out.println("Relevance judgment file loaded. [#q=" + relDocCount.size() + "]");
		}
		catch(IOException ex)
		{
			throw RankLibError.create("Error in APScorer::loadExternalRelevanceJudgment(): ", ex);
		}		
	}
	/**
	 * Compute Average Precision (AP) of the list. AP of a list is the average of precision evaluated at ranks where a relevant document 
	 * is observed. 
	 * @return AP of the list.
	 */
	public double score(RankList rl)
	{
		double ap = 0.0;
		int count = 0;
		for(int i=0;i<rl.size();i++)
		{
			if(rl.get(i).getLabel() > 0.0)//relevant
			{
				count++;
				ap += ((double)count)/(i+1);
			}
		}
		
		int rdCount = 0;
		if(relDocCount != null)
		{
			Integer it = relDocCount.get(rl.getID());
			if(it != null)
				rdCount = it;
		}
		else //no qrel-file specified, we can only use the #relevant-docs in the training file
			rdCount = count;
		
		if(rdCount==0)
			return 0.0;
		return ap / rdCount;
	}
	public String name()
	{
		return "MAP";
	}
	public double[][] swapChange(RankList rl)
	{
		//NOTE: Compute swap-change *IGNORING* K (consider the entire ranked list)
		int[] relCount = new int[rl.size()];
		int[] labels = new int[rl.size()];
		int count = 0;
		for(int i=0;i<rl.size();i++)
		{
			if(rl.get(i).getLabel() > 0)//relevant
			{
				labels[i] = 1;
				count++;
			}
			else
				labels[i] = 0;
			relCount[i] = count;
		}
		int rdCount = 0;//total number of relevant documents
		if(relDocCount != null)//if an external qrels file is specified
		{
			Integer it = relDocCount.get(rl.getID());
			if(it != null)
				rdCount = it;
		}
		else
			rdCount = count;

		double[][] changes = new double[rl.size()][];
		for(int i=0;i<rl.size();i++)
		{
			changes[i] = new double[rl.size()];
			Arrays.fill(changes[i], 0);
		}
		
		if(rdCount == 0 || count == 0)
			return changes;//all "0"
		
		for(int i=0;i<rl.size()-1;i++)
		{
			for(int j=i+1;j<rl.size();j++)
			{
				double change = 0;
				if(labels[i] != labels[j])
				{
					int diff = labels[j]-labels[i];
					change += ((double)((relCount[i]+diff)*labels[j] - relCount[i]*labels[i])) / (i+1);
					for(int k=i+1;k<=j-1;k++)
						if(labels[k] > 0)
							change += ((double)diff) / (k+1);
					change += ((double)(-relCount[j]*diff)) / (j+1);
					//It is equivalent to:  change += ((double)(relCount[j]*labels[i] - relCount[j]*labels[j])) / (j+1);
				}
				changes[j][i] = changes[i][j] = change/rdCount;				
			}
		}
		return changes;
	}
}
