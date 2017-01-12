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
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URLEncoder;

import edu.cmu.lti.oaqa.knn4qa.types.*;

/**
 * 
 * Creates DBPedia annotations.
 * 
 * @author Leonid Boytsov
 *
 */
public class DBPediaAnnot extends JCasAnnotator_ImplBase {
  private static final String RESOURCES_KEY = "Resources";

  private static final Logger logger = LoggerFactory.getLogger(DBPediaAnnot.class);
  
  //Create an instance of HttpClient.
  private static HttpClient mClient = new HttpClient();

  private static final String PARAM_SERVER_ADDR  = "ServerAddr";
  private static final String PARAM_CONF_THRESH  = "ConfThresh";
  private static final String ANNOT_PREFIX = "DBpedia:";
  
  public static final String DBPEDIA_TYPE = "dbpedia";
  
  @ConfigurationParameter(name = PARAM_SERVER_ADDR, mandatory = true)
  private String mServerAddr;
  
  @ConfigurationParameter(name = PARAM_CONF_THRESH, mandatory = true)
  private Float mConfThresh;  
  
  @Override
  public void initialize(UimaContext aContext)
  throws ResourceInitializationException {
    super.initialize(aContext);
  }
  
  @Override
  public void process(JCas aJCas) throws AnalysisEngineProcessException {
    String text = aJCas.getDocumentText(), resp;
    try {
      GetMethod getMethod = new GetMethod(mServerAddr + "/rest/annotate/?" +
          "confidence=" + mConfThresh
          //+ "&support=" + SUPPORT
          + "&text=" + URLEncoder.encode(text, "utf-8"));
      getMethod.addRequestHeader(new Header("Accept", "application/json"));

      resp = request(getMethod);
      
      JSONObject resultJSON = null;
      JSONArray  entities = null;
      
      try {
        resultJSON = new JSONObject(resp);
        if (!resultJSON.isNull(RESOURCES_KEY))
          entities = resultJSON.getJSONArray(RESOURCES_KEY);
      } catch (JSONException e) {
        throw new AnalysisEngineProcessException(
            new Exception("Received invalid response from DBpedia Spotlight API."));
      }
      
      if (entities != null)
      for(int i = 0; i < entities.length(); i++) {
        try {
          JSONObject entity = entities.getJSONObject(i);
          
          String types = entity.getString("@types");
          int    offset = Integer.parseInt(entity.getString("@offset"));
          String surfaceForm = entity.getString("@surfaceForm");
                            
          for (String t : types.split(",")) 
          if (t.startsWith(ANNOT_PREFIX)) {
            Entity e = new Entity(aJCas, offset, offset + surfaceForm.length());
            e.setEtype(DBPEDIA_TYPE);
            e.setLabel(t.substring(ANNOT_PREFIX.length()));
            e.addToIndexes();
          }
          
        } catch (JSONException e) {
          logger.error("JSON exception "+e);
          throw new AnalysisEngineProcessException(e);
        }
      }
      
    } catch (Exception e) {
      throw new AnalysisEngineProcessException(e);
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

}
