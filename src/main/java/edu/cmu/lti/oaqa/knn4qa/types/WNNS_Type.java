
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

/** Word net super senses.
 * Updated by JCasGen Thu Jan 12 23:52:10 EST 2017
 * @generated */
public class WNNS_Type extends Annotation_Type {
  /** @generated 
   * @return the generator for this type
   */
  @Override
  protected FSGenerator getFSGenerator() {return fsGenerator;}
  /** @generated */
  private final FSGenerator fsGenerator = 
    new FSGenerator() {
      public FeatureStructure createFS(int addr, CASImpl cas) {
  			 if (WNNS_Type.this.useExistingInstance) {
  			   // Return eq fs instance if already created
  		     FeatureStructure fs = WNNS_Type.this.jcas.getJfsFromCaddr(addr);
  		     if (null == fs) {
  		       fs = new WNNS(addr, WNNS_Type.this);
  			   WNNS_Type.this.jcas.putJfsFromCaddr(addr, fs);
  			   return fs;
  		     }
  		     return fs;
        } else return new WNNS(addr, WNNS_Type.this);
  	  }
    };
  /** @generated */
  @SuppressWarnings ("hiding")
  public final static int typeIndexID = WNNS.typeIndexID;
  /** @generated 
     @modifiable */
  @SuppressWarnings ("hiding")
  public final static boolean featOkTst = JCasRegistry.getFeatOkTst("edu.cmu.lti.oaqa.knn4qa.types.WNNS");
 
  /** @generated */
  final Feature casFeat_SuperSense;
  /** @generated */
  final int     casFeatCode_SuperSense;
  /** @generated
   * @param addr low level Feature Structure reference
   * @return the feature value 
   */ 
  public String getSuperSense(int addr) {
        if (featOkTst && casFeat_SuperSense == null)
      jcas.throwFeatMissing("SuperSense", "edu.cmu.lti.oaqa.knn4qa.types.WNNS");
    return ll_cas.ll_getStringValue(addr, casFeatCode_SuperSense);
  }
  /** @generated
   * @param addr low level Feature Structure reference
   * @param v value to set 
   */    
  public void setSuperSense(int addr, String v) {
        if (featOkTst && casFeat_SuperSense == null)
      jcas.throwFeatMissing("SuperSense", "edu.cmu.lti.oaqa.knn4qa.types.WNNS");
    ll_cas.ll_setStringValue(addr, casFeatCode_SuperSense, v);}
    
  



  /** initialize variables to correspond with Cas Type and Features
	 * @generated
	 * @param jcas JCas
	 * @param casType Type 
	 */
  public WNNS_Type(JCas jcas, Type casType) {
    super(jcas, casType);
    casImpl.getFSClassRegistry().addGeneratorForType((TypeImpl)this.casType, getFSGenerator());

 
    casFeat_SuperSense = jcas.getRequiredFeatureDE(casType, "SuperSense", "uima.cas.String", featOkTst);
    casFeatCode_SuperSense  = (null == casFeat_SuperSense) ? JCas.INVALID_FEATURE_CODE : ((FeatureImpl)casFeat_SuperSense).getCode();

  }
}



    