package edu.cmu.lti.oaqa.knn4qa.letor;

import java.util.ArrayList;
import java.util.Map;

import edu.cmu.lti.oaqa.knn4qa.memdb.DocEntry;
import edu.cmu.lti.oaqa.knn4qa.utils.VectorWrapper;
import no.uib.cipr.matrix.DenseVector;

/**
 * A single-field, single-score feature generator,
 * whose score can be computed (exactly or approximately)
 * as an inner product between two vectors.
 * 
 * @author Leonid Boytsov
 *
 */
public abstract class SingleFieldSingleScoreFeatExtractor extends SingleFieldFeatExtractor {

  @Override
  public abstract boolean isSparse();

  @Override
  public abstract int getDim();

  @Override
  public abstract String getFieldName();

  @Override
  public abstract String getName();

  @Override
  public abstract Map<String, DenseVector> 
        getFeatures(ArrayList<String> arrDocIds, Map<String, String> queryData) throws Exception;
  
  /**
   * This function produces a query and a document vector whose 
   * inner product is exactly or approximately equal to the only generated
   * feature value.
   * 
   * @param e a DocEntry object
   * 
   * @param isQuery true for queries and false for documents.
   * 
   * @return a possibly empty array of vector wrapper objects or null
   *         if the inner-product representation is not possible.
   * @throws Exception 
   */
  public abstract VectorWrapper getFeatInnerProdVector(DocEntry e, boolean isQuery) throws Exception;

  public int getFeatureQty() {
    return 1;
  }

}
