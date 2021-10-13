/*===============================================================================
 * Copyright (c) 2010-2012 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set 
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package ciir.umass.edu.learning.neuralnet;

import java.util.ArrayList;
import java.util.List;

/**
 * @author vdang
 * 
 * This class implements layers of neurons in neural networks.
 */
public class Layer {
	protected List<Neuron> neurons = null;
	
	public Layer(int size)
	{
		neurons = new ArrayList<Neuron>();
		for(int i=0;i<size;i++)
			neurons.add(new Neuron());
	}
	/**
	 * 
	 * @param size
	 * @param nType 0 for pair; 1 for list.
	 */
	public Layer(int size, int nType)
	{
		neurons = new ArrayList<Neuron>();
		for(int i=0;i<size;i++)
			if(nType == 0)
				neurons.add(new Neuron());
			else
				neurons.add(new ListNeuron());
	}
	public Neuron get(int k)
	{
		return neurons.get(k);
	}
	public int size()
	{
		return neurons.size();
	}
	
	/**
	 * Have all neurons in this layer compute its output
	 */
	public void computeOutput(int i)
	{
		for(int j=0;j<neurons.size();j++)
			neurons.get(j).computeOutput(i);
	}
	public void computeOutput()
	{
		for(int j=0;j<neurons.size();j++)
			neurons.get(j).computeOutput();
	}
	public void clearOutputs()
	{
		for(int i=0;i<neurons.size();i++)
			neurons.get(i).clearOutputs();
	}
	/**
	 * [Only for output layers] Compute delta for all neurons in the this (output) layer
	 * @param targetValues
	 */
	public void computeDelta(PropParameter param)
	{
		for(int i=0;i<neurons.size();i++)
			neurons.get(i).computeDelta(param);
	}
	/**
	 * Update delta from neurons in the previous layers
	 */
	public void updateDelta(PropParameter param)
	{
		for(int i=0;i<neurons.size();i++)
			neurons.get(i).updateDelta(param);
	}
	public void updateWeight(PropParameter param)
	{
		for(int i=0;i<neurons.size();i++)
			neurons.get(i).updateWeight(param);
	}
}
