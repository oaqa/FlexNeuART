
/* First created by JCasGen Thu Jan 12 13:26:20 EST 2017 */
package edu.cmu.lti.oaqa.knn4qa.types;

import org.apache.uima.jcas.JCas;
import org.apache.uima.jcas.JCasRegistry;
import org.apache.uima.cas.impl.CASImpl;
import org.apache.uima.cas.impl.FSGenerator;
import org.apache.uima.cas.FeatureStructure;
import org.apache.uima.cas.impl.TypeImpl;
import org.apache.uima.cas.Type;
import org.apache.uima.cas.impl.FeatureImpl;
import org.apache.uima.cas.Feature;
import org.apache.uima.jcas.tcas.Annotation_Type;

/** 
 * Updated by JCasGen Tue Jan 17 12:22:14 EST 2017
 * @generated */
public class TokenLemma_Type extends Annotation_Type {
  /** @generated 
   * @return the generator for this type
   */
  @Override
  protected FSGenerator getFSGenerator() {return fsGenerator;}
  /** @generated */
  private final FSGenerator fsGenerator = 
    new FSGenerator() {
      public FeatureStructure createFS(int addr, CASImpl cas) {
  			 if (TokenLemma_Type.this.useExistingInstance) {
  			   // Return eq fs instance if already created
  		     FeatureStructure fs = TokenLemma_Type.this.jcas.getJfsFromCaddr(addr);
  		     if (null == fs) {
  		       fs = new TokenLemma(addr, TokenLemma_Type.this);
  			   TokenLemma_Type.this.jcas.putJfsFromCaddr(addr, fs);
  			   return fs;
  		     }
  		     return fs;
        } else return new TokenLemma(addr, TokenLemma_Type.this);
  	  }
    };
  /** @generated */
  @SuppressWarnings ("hiding")
  public final static int typeIndexID = TokenLemma.typeIndexID;
  /** @generated 
     @modifiable */
  @SuppressWarnings ("hiding")
  public final static boolean featOkTst = JCasRegistry.getFeatOkTst("edu.cmu.lti.oaqa.knn4qa.types.TokenLemma");
 
  /** @generated */
  final Feature casFeat_Lemma;
  /** @generated */
  final int     casFeatCode_Lemma;
  /** @generated
   * @param addr low level Feature Structure reference
   * @return the feature value 
   */ 
  public String getLemma(int addr) {
        if (featOkTst && casFeat_Lemma == null)
      jcas.throwFeatMissing("Lemma", "edu.cmu.lti.oaqa.knn4qa.types.TokenLemma");
    return ll_cas.ll_getStringValue(addr, casFeatCode_Lemma);
  }
  /** @generated
   * @param addr low level Feature Structure reference
   * @param v value to set 
   */    
  public void setLemma(int addr, String v) {
        if (featOkTst && casFeat_Lemma == null)
      jcas.throwFeatMissing("Lemma", "edu.cmu.lti.oaqa.knn4qa.types.TokenLemma");
    ll_cas.ll_setStringValue(addr, casFeatCode_Lemma, v);}
    
  



  /** initialize variables to correspond with Cas Type and Features
	 * @generated
	 * @param jcas JCas
	 * @param casType Type 
	 */
  public TokenLemma_Type(JCas jcas, Type casType) {
    super(jcas, casType);
    casImpl.getFSClassRegistry().addGeneratorForType((TypeImpl)this.casType, getFSGenerator());

 
    casFeat_Lemma = jcas.getRequiredFeatureDE(casType, "Lemma", "uima.cas.String", featOkTst);
    casFeatCode_Lemma  = (null == casFeat_Lemma) ? JCas.INVALID_FEATURE_CODE : ((FeatureImpl)casFeat_Lemma).getCode();

  }
}



    