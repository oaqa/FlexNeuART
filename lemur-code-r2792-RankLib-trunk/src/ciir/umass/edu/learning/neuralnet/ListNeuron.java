/*===============================================================================
 * Copyright (c) 2010-2012 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set 
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package ciir.umass.edu.learning.neuralnet;

public class ListNeuron extends Neuron {
	
	protected double[] d1;
	protected double[] d2;	
	
	public void computeDelta(PropParameter param)
	{
		double sumLabelExp = 0;
		double sumScoreExp = 0;
		for(int i=0;i<outputs.size();i++)//outputs[i] ==> the output of the current neuron on the i-th document
		{
			sumLabelExp += Math.exp(param.labels[i]);
			sumScoreExp += Math.exp(outputs.get(i));
		}

		d1 = new double[outputs.size()];
		d2 = new double[outputs.size()];
		for(int i=0;i<outputs.size();i++)
		{
			d1[i] = Math.exp(param.labels[i])/sumLabelExp;
			d2[i] = Math.exp(outputs.get(i))/ sumScoreExp;
		}
	}
	public void updateWeight(PropParameter param)
	{
		Synapse s = null;
		for(int k=0;k<inLinks.size();k++)
		{
			s = inLinks.get(k);
			double dw = 0;
			for(int l=0;l<d1.length;l++)
				dw += (d1[l] - d2[l]) * s.getSource().getOutput(l);
			
			dw *= learningRate;
			s.setWeightAdjustment(dw);
			s.updateWeight();
		}
	}
}
