/*
 *  Copyright 2017 Carnegie Mellon University
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

package edu.cmu.lti.oaqa.knn4qa.annotators;

import java.util.ArrayList;

import org.apache.uima.analysis_engine.AnalysisEngine;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This class works as a JCAS factory.
 * 
 * <p>The difference from UIMAFit is that
 * this factory keeps a pool of JCASes, where you should return a JCAS
 * instead of calling release(). The latter somehow does not free the
 * memory IMHO.</p>
 * 
 * @author Leonid Boytsov
 *
 */
public class JCasFactory {
  private static final Logger logger = LoggerFactory.getLogger(JCasFactory.class);
  
  public JCasFactory(AnalysisEngine engine) { 
    mEngine = engine; 
  }
  
  public JCas borrowJCas() throws ResourceInitializationException {
    synchronized (this) {
      if (!mJCasList.isEmpty()) {
        int last = mJCasList.size() - 1;
        JCas res = mJCasList.get(last);
        mJCasList.remove(last);
        // Reset here just in case, but it is also reset
        // in the returnJCas function
        res.reset();      
        return res;
      }
      ++mCasAllocQty;
      logger.info("Total number of allocations: " + mCasAllocQty);
      return mEngine.newJCas();
    }
  }
  
  public void returnJCas(JCas jcas) {
    synchronized (this) {
      jcas.reset();
      mJCasList.add(jcas);
    }
  }
  
  public void destroy() {
    mEngine.destroy();
  }
  
  private long             mCasAllocQty = 0;
  private AnalysisEngine   mEngine;
  private ArrayList<JCas>  mJCasList = new ArrayList<JCas>();     
}
