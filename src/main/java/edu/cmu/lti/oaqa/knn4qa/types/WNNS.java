

/* First created by JCasGen Mon Nov 30 13:17:14 EST 2015 */
package edu.cmu.lti.oaqa.knn4qa.types;

import org.apache.uima.jcas.JCas; 
import org.apache.uima.jcas.JCasRegistry;
import org.apache.uima.jcas.cas.TOP_Type;

import org.apache.uima.jcas.tcas.Annotation;


/** Word net super senses.
 * Updated by JCasGen Mon Nov 30 13:17:14 EST 2015
 * XML source: /home/leo/SourceTreeGit/leo_struct_ir_qa/src/main/resources/types/typeSystemDescriptor.xml
 * @generated */
public class WNNS extends Annotation {
  /** @generated
   * @ordered 
   */
  @SuppressWarnings ("hiding")
  public final static int typeIndexID = JCasRegistry.register(WNNS.class);
  /** @generated
   * @ordered 
   */
  @SuppressWarnings ("hiding")
  public final static int type = typeIndexID;
  /** @generated
   * @return index of the type  
   */
  @Override
  public              int getTypeIndexID() {return typeIndexID;}
 
  /** Never called.  Disable default constructor
   * @generated */
  protected WNNS() {/* intentionally empty block */}
    
  /** Internal - constructor used by generator 
   * @generated
   * @param addr low level Feature Structure reference
   * @param type the type of this Feature Structure 
   */
  public WNNS(int addr, TOP_Type type) {
    super(addr, type);
    readObject();
  }
  
  /** @generated
   * @param jcas JCas to which this Feature Structure belongs 
   */
  public WNNS(JCas jcas) {
    super(jcas);
    readObject();   
  } 

  /** @generated
   * @param jcas JCas to which this Feature Structure belongs
   * @param begin offset to the begin spot in the SofA
   * @param end offset to the end spot in the SofA 
  */  
  public WNNS(JCas jcas, int begin, int end) {
    super(jcas);
    setBegin(begin);
    setEnd(end);
    readObject();
  }   

  /** 
   * <!-- begin-user-doc -->
   * Write your own initialization here
   * <!-- end-user-doc -->
   *
   * @generated modifiable 
   */
  private void readObject() {/*default - does nothing empty block */}
     
 
    
  //*--------------*
  //* Feature: SuperSense

  /** getter for SuperSense - gets 
   * @generated
   * @return value of the feature 
   */
  public String getSuperSense() {
    if (WNNS_Type.featOkTst && ((WNNS_Type)jcasType).casFeat_SuperSense == null)
      jcasType.jcas.throwFeatMissing("SuperSense", "edu.cmu.lti.oaqa.knn4qa.types.WNNS");
    return jcasType.ll_cas.ll_getStringValue(addr, ((WNNS_Type)jcasType).casFeatCode_SuperSense);}
    
  /** setter for SuperSense - sets  
   * @generated
   * @param v value to set into the feature 
   */
  public void setSuperSense(String v) {
    if (WNNS_Type.featOkTst && ((WNNS_Type)jcasType).casFeat_SuperSense == null)
      jcasType.jcas.throwFeatMissing("SuperSense", "edu.cmu.lti.oaqa.knn4qa.types.WNNS");
    jcasType.ll_cas.ll_setStringValue(addr, ((WNNS_Type)jcasType).casFeatCode_SuperSense, v);}    
  }

    