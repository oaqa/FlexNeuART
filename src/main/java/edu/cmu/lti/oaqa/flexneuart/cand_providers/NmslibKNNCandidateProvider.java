/*
 *  Copyright 2014+ Carnegie Mellon University
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
package edu.cmu.lti.oaqa.flexneuart.cand_providers;

import java.io.ByteArrayOutputStream;
import java.util.*;

import org.apache.thrift.protocol.TBinaryProtocol;
import org.apache.thrift.transport.TSocket;
import org.apache.thrift.transport.TTransport;

import com.google.common.base.Splitter;

import edu.cmu.lti.oaqa.flexneuart.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.flexneuart.letor.CompositeFeatureExtractor;
import edu.cmu.lti.oaqa.flexneuart.letor.FeatExtrResourceManager;
import edu.cmu.lti.oaqa.flexneuart.letor.SingleFieldFeatExtractor;
import edu.cmu.lti.oaqa.flexneuart.letor.SingleFieldInnerProdFeatExtractor;
import edu.cmu.lti.oaqa.flexneuart.utils.DataEntryFields;
import edu.cmu.lti.oaqa.flexneuart.utils.VectorUtils;
import edu.cmu.lti.oaqa.nmslib.QueryService;
import edu.cmu.lti.oaqa.nmslib.ReplyEntry;
import edu.cmu.lti.oaqa.nmslib.QueryService.Client;
import no.uib.cipr.matrix.DenseVector;

/**
 * NMSLIB-based KNN to return k-closest entries. It currently supports
 * only two types of spaces, which exist only the branch 
 * https://github.com/nmslib/nmslib/tree/nmslib4a_postdoc: 
 * 
 * 1) a dense-sparse fusion space, which stores a mix of sparse and dense vectors.
 * 2) a sparse space, where all dense and sparse vector components are interleaved.
 * 
 * @author Leonid Boytsov
 *
 */
public class NmslibKNNCandidateProvider  extends CandidateProvider {
  
  public final static String EXTRACTOR_TYPE_PARAM = "extrType";
  public final static String SPARSE_INTERLEAVE_PARAM = "sparseInterleave";
  
  Splitter splitOnColon = Splitter.on(':');	
  final private Client 			                  mKNNClient;
  final private FeatExtrResourceManager       mResourceManager;
  final private SingleFieldInnerProdFeatExtractor  mCompExtractors[];
  final private ForwardIndex                  mCompIndices[];
  final int                                   mFeatExtrQty;
  final DenseVector                           mCompQueryWeights;

  /**
   * Constructor.
   * 
   * @param knnServiceURL       an address/port for the NMSLIB server.
   * @param resourceManager     a resource manager.
   * @param addConf             additional (but mandatory) parameters
   * 
   * @throws Exception
   */
  public NmslibKNNCandidateProvider(String knnServiceURL, 
                                    FeatExtrResourceManager resourceManager,
                                    CandProvAddConfig addConf) throws Exception {

    String host = null;
    int    port = -1;
    
    if (addConf == null) {
      throw new Exception(this.getClass().getName() + " requres a non-null configuration");
    }
    
    String extrType = addConf.getReqParamStr(EXTRACTOR_TYPE_PARAM);
    boolean sparseInterleave = addConf.getParamBool(SPARSE_INTERLEAVE_PARAM);
    
    CompositeFeatureExtractor featExtr = new CompositeFeatureExtractor(resourceManager, extrType);
    
    mResourceManager = resourceManager;   
    
    SingleFieldFeatExtractor[] allExtractors = featExtr.getCompExtr();
    mFeatExtrQty = allExtractors.length;
    
    // Potentially we can use these weights to re-mix component weights
    // differently (from indexing mix) at query time. However,
    // currently we do not do it
    mCompQueryWeights = sparseInterleave ? VectorUtils.fill(1, mFeatExtrQty) : null;
    
    int featExtrQty = allExtractors.length;
    mCompExtractors = new SingleFieldInnerProdFeatExtractor[featExtrQty];
    
    for (int i = 0; i < featExtrQty; ++i) {
      if (!(allExtractors[i] instanceof SingleFieldInnerProdFeatExtractor)) {
        throw new Exception("Sub-extractor # " + (i+1) + " (" + allExtractors[i].getName() 
                            +") doesn't support export to NMSLIB");
      }
      mCompExtractors[i] = (SingleFieldInnerProdFeatExtractor)allExtractors[i];
    }

    mCompIndices  = new ForwardIndex[mFeatExtrQty];

    for (int i = 0; i < mFeatExtrQty; ++i) {
      mCompIndices[i] = mResourceManager.getFwdIndex(mCompExtractors[i].getIndexFieldName());
    }

    int part = 0;
    for (String s : splitOnColon.split(knnServiceURL)) {
      if (0 == part) {
        host = s;
      } else if (1 == part) {
        try {
          port = Integer.parseInt(s);
        } catch (NumberFormatException e) {
          throw new RuntimeException("Invalid port in the service address '" + knnServiceURL + "'");
        }
      } else {
        throw new RuntimeException("Extra colon in the service address in '" + knnServiceURL + "'");
      }
      ++part;
    }

    if (part != 2) {
      throw new Exception("Invalid format of the service address in '" + knnServiceURL + "'");
    }

    TTransport knnServiceTransp = new TSocket(host, port);
    knnServiceTransp.open();
    mKNNClient = new QueryService.Client(new TBinaryProtocol(knnServiceTransp));
  }

  @Override
  public String getName() {
    return this.getClass().getName();
  }

  /*
   * getCandidates is not thread-safe and each thread need to use its own Transport and Client.
   * From this URL: http://comments.gmane.org/gmane.comp.lib.thrift.user/2704
   * 
   * The Thrift transport layer is not thread-safe. It is essentially a wrapper on a socket.

        You can't interleave writing things to a single socket from multiple threads without locking. You also
        don't know what order the responses will come back in. Each thread is effectively calling read(). To have
        this work in a multi-threaded environment would require another layer of abstraction that parceled out
        responses on the socket and determined which data should go to which thread. This would be less efficient
        in the common case of a single transport per thread.
   */  
  @Override
  public boolean isThreadSafe() { return false; }   	

  @Override
  public CandidateInfo getCandidates(int queryNum, DataEntryFields queryFields, int maxQty) throws Exception {
       
    String queryId = queryFields.mEntryId;
    
    if (queryId == null) {
      throw new Exception("No query ID");
    }
    
    ByteArrayOutputStream  out = new ByteArrayOutputStream();
    
    if (null == mCompQueryWeights) {
      VectorUtils.writeInnerProdQueryVecsToNMSLIBStream(
                                                  queryFields, 
                                                  mCompIndices, 
                                                  mCompExtractors, 
                                                  out);
    } else {
      VectorUtils.writeInterleavedInnerProdQueryVectToNMSLIBStream(
                                                            queryFields, 
                                                            mCompIndices, 
                                                            mCompExtractors, 
                                                            mCompQueryWeights, 
                                                            out);
    }
     
    java.nio.ByteBuffer  queryObj = java.nio.ByteBuffer.wrap(out.toByteArray());

    // queyrObj MUST be byteBuffer
    List<ReplyEntry> clientRepl = mKNNClient.knnQuery(maxQty, queryObj, true, false);

    CandidateEntry[] res = new CandidateEntry[Math.min(clientRepl.size(), maxQty)];

    int ind = 0;

    HashSet<String> seen = new HashSet<String>(clientRepl.size());

    for (ReplyEntry e : clientRepl) {		  
      if (ind >= res.length) break;
      String externId = e.getExternId();
      if (seen.contains(externId)) continue;
      seen.add(externId);
      res[ind++] = new CandidateEntry(externId, (float) -e.getDist());
    }		

    // If there were duplicates there would be some NULL entries in the res array, let's get rid of them
    if (ind < res.length)
      res = Arrays.copyOf(res, ind);

    return new CandidateInfo(res);
  }

}
