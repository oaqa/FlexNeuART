package edu.cmu.lti.oaqa.knn4qa.annotators;

import org.apache.uima.UimaContext;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;
import org.apache.uima.fit.component.JCasAnnotator_ImplBase;
import org.apache.uima.fit.descriptor.ConfigurationParameter;
import org.apache.commons.httpclient.*;
import org.apache.commons.httpclient.methods.GetMethod;
import org.apache.commons.httpclient.params.HttpMethodParams;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URLEncoder;
import java.util.*;

import edu.cmu.lti.oaqa.knn4qa.dbpedia.OntologyReader;
import edu.cmu.lti.oaqa.knn4qa.types.*;

/**
 * 
 * Creates DBPedia annotations.
 * 
 * <p>Note that in very rare cases, DBPedia SpotLight fails for no apparent reason. For example,
 * instead of a valid response, I get an error that starts likes this:</p>
 * <pre>
 * org.dbpedia.spotlight.exceptions.OutputException: Error converting XML to JSON.Error converting XML to JSON.org.dbpedia.spotlight.web.rest.OutputManager.xml2json(OutputManager.java:235)
 * </pre>
 * <p>Because such errors are extremely infrequent, we prefer to ignore them and print the total error count in the end so
 * leaving the judgment (as to how bad this is) to the user.
 * </p>
 * 
 * @author Leonid Boytsov
 *
 */
public class DBPediaAnnot extends JCasAnnotator_ImplBase {
 
  private static final String RESOURCES_KEY = "Resources";

  private static final Logger logger = LoggerFactory.getLogger(DBPediaAnnot.class);
  
  //Create an instance of HttpClient.
  private HttpClient mClient = new HttpClient();

  private static final String PARAM_SERVER_ADDR  = "ServerAddr";
  private static final String PARAM_CONF_THRESH  = "ConfThresh";
  private static final String PARAM_ONTOLOGY_FILE= "OntologyFile";
  private static final String PARAM_MAX_FAILURE_QTY = "MaxFailureQty";
  
  private static final String ANNOT_PREFIX       = "DBpedia:";
   
  
  public static final String DBPEDIA_TYPE = "dbpedia";
  
  @ConfigurationParameter(name = PARAM_SERVER_ADDR, mandatory = true)
  private String mServerAddr;
  
  @ConfigurationParameter(name = PARAM_CONF_THRESH, mandatory = true)
  private Float mConfThresh;  

  @ConfigurationParameter(name = PARAM_ONTOLOGY_FILE, mandatory = true)
  private String mOntologyFile;    

  @ConfigurationParameter(name = PARAM_MAX_FAILURE_QTY, mandatory = true)
  private Integer mMaxFailureQty;    
  
  private OntologyReader mDBPediaOnto;
  
  private static int mErrorQty = 0;
  
  @Override
  public void initialize(UimaContext aContext)
  throws ResourceInitializationException {
    super.initialize(aContext);
    
    try {
      mDBPediaOnto = new OntologyReader(mOntologyFile);
    } catch (FileNotFoundException e) {
      e.printStackTrace();
      throw new ResourceInitializationException(e);
    }
  }
  
  /**
   * Remove annotations for which there is a child annotation in the list, e.g.,
   * remove Person if Athlete is present.
   * 
   * @param annotList   an initial list of annotations
   * @return a culled list of annotations
   */
  ArrayList<String> removeParents(ArrayList<String> annotList) {
    ArrayList<String> resOld = new ArrayList<String>();        
    ArrayList<String> resNew = new ArrayList<String>();

    // Before the loop start, the new values array contains the complete list of annotations
    for (String s: annotList) resNew.add(s);
    
    do {
      // In the beginning of the loop, we copy new values (resNew array) to the resOld array
      // Note that before loop starts, all annotation are copied to the new values array (resNew)
      resOld.clear();
      for (String s : resNew) resOld.add(s);
      // Now we remove all values from the new array
      resNew.clear();
      // These loops have quadratic cost, but remember that 
      // the number of elements to compare is small, so that
      // quadratic is not slow here
      for (int i1 = 0; i1 < resOld.size(); ++i1) {
        boolean isSuper = false; // True, if resOld.get(i1) is a super class of another annotation
        
        for (int i2 = 0; i2 < resOld.size(); ++i2) 
        if (i1 != i2 && mDBPediaOnto.isSubClass(resOld.get(i2), resOld.get(i1))) {
          isSuper = true; break;
        }
        if (!isSuper) resNew.add(resOld.get(i1));
      }
    } while (resNew.size() < resOld.size());
    
    
    return resOld;
  }
  
  @Override
  public void process(JCas aJCas) throws AnalysisEngineProcessException {
    String text = aJCas.getDocumentText(), resp;
    
    if (mErrorQty > mMaxFailureQty) {
      throw new AnalysisEngineProcessException(
                new Exception("The number of failures exceeded the threshold " + mMaxFailureQty));
    }
    
    synchronized (this.getClass()) {
      ++mErrorQty; // we will later decrease error counter, unless the function
      // terminates preliminarly
    }
    
    try {
      GetMethod getMethod = new GetMethod(mServerAddr + "/rest/annotate/?" +
          "confidence=" + mConfThresh
          //+ "&support=" + SUPPORT
          + "&text=" + URLEncoder.encode(text, "utf-8"));
      getMethod.addRequestHeader(new Header("Accept", "application/json"));

      try {
        resp = request(getMethod);
      } catch (Exception e) {
        logger.error("Error sending a request to DBPedia SpotLight: " + e);
        return;
      }
      
      JSONObject resultJSON = null;
      JSONArray  entities = null;
      
      try {
        resultJSON = new JSONObject(resp);
        if (!resultJSON.isNull(RESOURCES_KEY))
          entities = resultJSON.getJSONArray(RESOURCES_KEY);
      } catch (JSONException e) {
        //e.printStackTrace();
        logger.error("Received the following invalid response from DBpedia Spotlight API: '" + resp + "'");
        return;
      }
      
      if (entities != null)
      for(int i = 0; i < entities.length(); i++) {
        try {
          JSONObject entity = entities.getJSONObject(i);
          
          String types = entity.getString("@types");
          int    offset = Integer.parseInt(entity.getString("@offset"));
          String surfaceForm = entity.getString("@surfaceForm");
                            
          ArrayList<String> annotList = new ArrayList<String>();
          
          for (String t : types.split(",")) 
          if (t.startsWith(ANNOT_PREFIX)) {
            annotList.add(t.substring(ANNOT_PREFIX.length()));
          }
          
          for (String t : removeParents(annotList)) {
            Entity e = new Entity(aJCas, offset, offset + surfaceForm.length());
            e.setEtype(DBPEDIA_TYPE);
            e.setLabel(t);
            e.addToIndexes();
          }
          
        } catch (JSONException e) {
          //e.printStackTrace();
          logger.error("JSON exception "+e);
          return;
        }
      }
      
    } catch (Exception e) {
      logger.error("Exception: " + e);
      return;
    }
    
    synchronized (this.getClass()) {
      --mErrorQty;
    }

  }
  
  // This is a thoroughly rewritten DBPedia's AnnotationClient.request method
  public String request(HttpMethod method) throws Exception {
    String response = null;

    // Provide custom retry handler is necessary
    method.getParams().setParameter(HttpMethodParams.RETRY_HANDLER,
            new DefaultHttpMethodRetryHandler(3, false));

    try {
        // Execute the method.
        int statusCode = mClient.executeMethod(method);

        if (statusCode != HttpStatus.SC_OK) {
            logger.error("Method failed: " + method.getStatusLine());
        }

        StringBuffer sb = new StringBuffer();
        BufferedReader buffResp = new BufferedReader(new InputStreamReader(method.getResponseBodyAsStream()));
        String  line;
        while ((line = buffResp.readLine()) != null) {
          sb.append(line);
        }        
        response = sb.toString();        
    } catch (HttpException e) {
        logger.error("Fatal protocol violation: " + e.getMessage());
        throw new Exception("Protocol error executing HTTP request.",e);
    } catch (IOException e) {
        logger.error("Fatal transport error: " + e.getMessage());
        logger.error(method.getQueryString());
        throw new Exception("Transport error executing HTTP request.",e);
    } finally {
        // Release the connection.
        method.releaseConnection();
    }
    return response;
  }
  
  @Override
  public void collectionProcessComplete() throws AnalysisEngineProcessException {
    logger.info("The pipeline complete, annotator " + this.getClass().getCanonicalName() + 
                " failed " + mErrorQty + " times");
  }

}
