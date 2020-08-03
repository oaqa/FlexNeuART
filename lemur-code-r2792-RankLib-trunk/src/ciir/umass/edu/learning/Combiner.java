/*===============================================================================
 * Copyright (c) 2010-2012 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set 
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package ciir.umass.edu.learning;

import ciir.umass.edu.learning.tree.Ensemble;
import ciir.umass.edu.learning.tree.RFRanker;
import ciir.umass.edu.utilities.FileUtils;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;

public class Combiner {
	public static void main(String[] args)
	{
		Combiner c = new Combiner();
		c.combine(args[0], args[1]);
	}
	public void combine(String directory, String outputFile)
	{
		RankerFactory rf = new RankerFactory();
		String[] fns = FileUtils.getAllFiles(directory);
		BufferedWriter out = null;
		try{
			out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFile), "ASCII"));
			out.write("## " + (new RFRanker()).name() + "\n");
			for(int i=0;i<fns.length;i++)
			{
				if(fns[i].indexOf(".progress") != -1)
					continue;
				String fn = directory + fns[i];
				RFRanker r = (RFRanker)rf.loadRankerFromFile(fn);
				Ensemble en = r.getEnsembles()[0];
				out.write(en.toString());
			}
			out.close();
		}
		catch(Exception e)
		{
			System.out.println("Error in Combiner::combine(): " + e.toString());
		}
	}
}
