package edu.cmu.lti.oaqa.flexneuart.letor;

import java.util.Map;

import edu.cmu.lti.oaqa.flexneuart.cand_providers.CandidateEntry;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.DocEntryParsed;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.flexneuart.resources.RestrictedJsonConfig;
import edu.cmu.lti.oaqa.flexneuart.resources.ResourceManager;
import edu.cmu.lti.oaqa.flexneuart.utils.DataEntryFields;
import edu.cmu.lti.oaqa.flexneuart.utils.VectorWrapper;
import no.uib.cipr.matrix.DenseVector;

/**
 * A single-field (or rather a single pair of query and index fields), 
 * single-score feature generator, whose score can be computed (exactly or approximately)
 * as an inner product between two vectors. Note, again,
 * that query and index fields can be different.
 * 
 * @author Leonid Boytsov
 *
 */
public abstract class SingleFieldInnerProdFeatExtractor extends SingleFieldFeatExtractor {

  public SingleFieldInnerProdFeatExtractor(ResourceManager resMngr, RestrictedJsonConfig conf) throws Exception {
    super(resMngr, conf);
  }

  /**
   * @return true if generates a sparse feature vector.
   */
  public abstract boolean isSparse();

  public abstract int getDim();
  
  @Override
  public abstract String getName();

  @Override
  public abstract Map<String, DenseVector> 
                  getFeatures(CandidateEntry cands[], DataEntryFields queryFields) throws Exception;
  
  /**
   * Generates query vector that together with respective document vector (possibly
   * approximately) reproduce feature values via inner-product computation. 
   * An implementation may be missing for some types of extractors.
   * 
   * @param query   a query string.
   * 
   * @return a vector wrapper object or null if the inner-product representation is not possible.
   * @throws Exception 
   */
  public VectorWrapper getFeatInnerProdQueryVector(String query) throws Exception {
    return null;
  }
  
  /**
   * Generates document vector that together with respective query vector (possibly
   * approximately) reproduce feature values via inner-product computation. 
   * An implementation may be missing for some types of extractors.
   * 
   * @param query   a document string.
   * 
   * @return a vector wrapper object or null if the inner-product representation is not possible.
   * @throws Exception 
   */
  public VectorWrapper getFeatInnerProdDocVector(String doc) throws Exception {
    return null;
  }
  
  /**
   * Generates query vector that together with document vectors (possibly
   * approximately) reproduce feature values via inner-product computation. 
   * An implementation may be missing for some types of extractors.
   * 
   * @param query    a parsed query
   * 
   * @return a vector wrapper object or null if the inner-product representation is not possible.
   * @throws Exception 
   */
  public VectorWrapper getFeatInnerProdQueryVector(DocEntryParsed query) throws Exception {
    return null;
  }
  
  public VectorWrapper getFeatInnerProdDocVector(DocEntryParsed doc) throws Exception {
    return null;
  }
  
  /**
   * Generates query vector that together with document vectors (possibly
   * approximately) reproduce feature values via inner-product computation. 
   * An implementation may be missing for some types of extractors.
   * 
   * @param query    a parsed query
   * 
   * @return a vector wrapper object or null if the inner-product representation is not possible.
   * @throws Exception 
   */
  public VectorWrapper getFeatInnerProdQueryVector(byte[] query) throws Exception {
    return null;
  }
  
  public VectorWrapper getFeatInnerProdDocVector(byte[] doc) throws Exception {
    return null;
  }
  
  /**
   * Generate a batch of document feature vectors that together with query vectors (possibly
   * approximately) reproduce feature values via inner-product computation.
   * 
   * <p>This top-level function merely calls {@link #getFeatDocInnerProdVector(DocEntryParsed, boolean)}
   * or {@link #getFeatDocInnerProdVector(String, boolean)}.</p>
   * 
   * <p>But child classes can override it and provide a more efficient batched version, e.g., by
   * processing it on a GPU.</p>
   * 
   * @param fwdIndx     a forward index.
   * @param docIds      an array of document IDs.
   * @return
   * @throws Exception
   */
  public VectorWrapper[] getFeatInnerProdDocVectorBatch(ForwardIndex fwdIndx, String docIds[]) throws Exception {
    int qty = docIds.length;
    VectorWrapper[] res = new VectorWrapper[qty];
    
    for (int i = 0; i < qty; i++) {
      String did = docIds[i];
      if (fwdIndx.isTextRaw()) {
        String docEntryRaw = fwdIndx.getDocEntryTextRaw(did);
        if (docEntryRaw == null) {
          throw new Exception("Inconsistent data or bug: can't find document with id ='" + did + "'");
        }
        res[i] = getFeatInnerProdDocVector(docEntryRaw);
      } else if (fwdIndx.isParsed()) {
        DocEntryParsed docEntryParsed = fwdIndx.getDocEntryParsed(did);
        if (docEntryParsed == null) {
          throw new Exception("Inconsistent data or bug: can't find document with id ='" + did + "'");
        }
        res[i] = getFeatInnerProdDocVector(docEntryParsed);
      } else if (fwdIndx.isBinary()) {
        byte [] docEntryBinary = fwdIndx.getDocEntryBinary(did);
        if (docEntryBinary == null) {
          throw new Exception("Inconsistent data or bug: can't find document with id ='" + did + "'");
        }
        res[i] = getFeatInnerProdDocVector(docEntryBinary);
      } else {
        throw new Exception("Unsupported index type: " + fwdIndx.getIndexFieldType());
      }
    }
    
    return res;
  }
  
  public int getFeatureQty() {
    return 1;
  }

}
