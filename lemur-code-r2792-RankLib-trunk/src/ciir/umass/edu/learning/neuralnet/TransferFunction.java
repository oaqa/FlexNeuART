/*===============================================================================
 * Copyright (c) 2010-2012 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set 
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package ciir.umass.edu.learning.neuralnet;

/**
 * @author vdang
 * 
 * This is the abstract class for implementing transfer functions for neuralnet.
 */
public interface TransferFunction {
	public double compute(double x);
	public double computeDerivative(double x);
}
