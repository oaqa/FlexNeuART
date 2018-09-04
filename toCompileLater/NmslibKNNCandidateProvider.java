package edu.cmu.lti.oaqa.knn4qa.cand_providers;

import java.util.*;

import org.apache.thrift.protocol.TBinaryProtocol;
import org.apache.thrift.transport.TSocket;
import org.apache.thrift.transport.TTransport;

import com.google.common.base.Splitter;

import edu.cmu.lti.oaqa.similarity.QueryService;
import edu.cmu.lti.oaqa.similarity.QueryService.Client;
import edu.cmu.lti.oaqa.similarity.ReplyEntry;

/**
 * NMSLIB-based KNN to return k-closest entries.
 * 
 * @author Leonid Boytsov
 *
 */
public class NmslibKNNCandidateProvider  extends CandidateProvider {
	Splitter splitOnColon = Splitter.on(':');	
	final private NmslibQueryGenerator 	mQueryGen;
	final private Client 							  mKNNClient;

	public NmslibKNNCandidateProvider(String knnServiceURL, NmslibQueryGenerator queryGen) throws Exception {
		mQueryGen = queryGen;
   	String host = null;
  	int    port = -1;
  	
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
	public CandidateInfo getCandidates(int queryNum, Map<String, String> queryData, int maxQty) throws Exception {
		String queryObjStr = mQueryGen.getStrObjForKNNService(queryData);
		
		List<ReplyEntry> clientRepl = mKNNClient.knnQuery(maxQty, queryObjStr, true, false);
		
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
