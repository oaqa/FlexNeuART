

/* First created by JCasGen Thu Jan 12 13:26:20 EST 2017 */
package edu.cmu.lti.oaqa.knn4qa.types;

import org.apache.uima.jcas.JCas; 
import org.apache.uima.jcas.JCasRegistry;
import org.apache.uima.jcas.cas.TOP_Type;

import org.apache.uima.jcas.tcas.Annotation;


/** A description of one question.
 * Updated by JCasGen Thu Jan 12 13:26:20 EST 2017
 * XML source: /home/leo/SourceTreeGit/knn4qa_oqaqa/src/main/resources/types/typeSystemDescriptor.xml
 * @generated */
public class Question extends Annotation {
  /** @generated
   * @ordered 
   */
  @SuppressWarnings ("hiding")
  public final static int typeIndexID = JCasRegistry.register(Question.class);
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
  protected Question() {/* intentionally empty block */}
    
  /** Internal - constructor used by generator 
   * @generated
   * @param addr low level Feature Structure reference
   * @param type the type of this Feature Structure 
   */
  public Question(int addr, TOP_Type type) {
    super(addr, type);
    readObject();
  }
  
  /** @generated
   * @param jcas JCas to which this Feature Structure belongs 
   */
  public Question(JCas jcas) {
    super(jcas);
    readObject();   
  } 

  /** @generated
   * @param jcas JCas to which this Feature Structure belongs
   * @param begin offset to the begin spot in the SofA
   * @param end offset to the end spot in the SofA 
  */  
  public Question(JCas jcas, int begin, int end) {
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
  //* Feature: uri

  /** getter for uri - gets A unique question resource identifier
   * @generated
   * @return value of the feature 
   */
  public String getUri() {
    if (Question_Type.featOkTst && ((Question_Type)jcasType).casFeat_uri == null)
      jcasType.jcas.throwFeatMissing("uri", "edu.cmu.lti.oaqa.knn4qa.types.Question");
    return jcasType.ll_cas.ll_getStringValue(addr, ((Question_Type)jcasType).casFeatCode_uri);}
    
  /** setter for uri - sets A unique question resource identifier 
   * @generated
   * @param v value to set into the feature 
   */
  public void setUri(String v) {
    if (Question_Type.featOkTst && ((Question_Type)jcasType).casFeat_uri == null)
      jcasType.jcas.throwFeatMissing("uri", "edu.cmu.lti.oaqa.knn4qa.types.Question");
    jcasType.ll_cas.ll_setStringValue(addr, ((Question_Type)jcasType).casFeatCode_uri, v);}    
   
    
  //*--------------*
  //* Feature: bestAnswId

  /** getter for bestAnswId - gets 
   * @generated
   * @return value of the feature 
   */
  public int getBestAnswId() {
    if (Question_Type.featOkTst && ((Question_Type)jcasType).casFeat_bestAnswId == null)
      jcasType.jcas.throwFeatMissing("bestAnswId", "edu.cmu.lti.oaqa.knn4qa.types.Question");
    return jcasType.ll_cas.ll_getIntValue(addr, ((Question_Type)jcasType).casFeatCode_bestAnswId);}
    
  /** setter for bestAnswId - sets  
   * @generated
   * @param v value to set into the feature 
   */
  public void setBestAnswId(int v) {
    if (Question_Type.featOkTst && ((Question_Type)jcasType).casFeat_bestAnswId == null)
      jcasType.jcas.throwFeatMissing("bestAnswId", "edu.cmu.lti.oaqa.knn4qa.types.Question");
    jcasType.ll_cas.ll_setIntValue(addr, ((Question_Type)jcasType).casFeatCode_bestAnswId, v);}    
  }

    