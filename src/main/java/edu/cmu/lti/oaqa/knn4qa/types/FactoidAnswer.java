

/* First created by JCasGen Tue Jan 17 11:16:34 EST 2017 */
package edu.cmu.lti.oaqa.knn4qa.types;

import org.apache.uima.jcas.JCas; 
import org.apache.uima.jcas.JCasRegistry;
import org.apache.uima.jcas.cas.TOP_Type;

import org.apache.uima.jcas.tcas.Annotation;


/** An answer to a factoid question
 * Updated by JCasGen Tue Jan 17 12:22:14 EST 2017
 * XML source: /home/leo/SourceTreeGit/knn4qa_oqaqa/src/main/resources/types/typeSystemDescriptor.xml
 * @generated */
public class FactoidAnswer extends Annotation {
  /** @generated
   * @ordered 
   */
  @SuppressWarnings ("hiding")
  public final static int typeIndexID = JCasRegistry.register(FactoidAnswer.class);
  /** @generated
   * @ordered 
   */
  @SuppressWarnings ("hiding")
  public final static int type = typeIndexID;
  /** @generated  */
  @Override
  public              int getTypeIndexID() {return typeIndexID;}
 
  /** Never called.  Disable default constructor
   * @generated */
  protected FactoidAnswer() {/* intentionally empty block */}
    
  /** Internal - constructor used by generator 
   * @generated */
  public FactoidAnswer(int addr, TOP_Type type) {
    super(addr, type);
    readObject();
  }
  
  /** @generated */
  public FactoidAnswer(JCas jcas) {
    super(jcas);
    readObject();   
  } 

  /** @generated */  
  public FactoidAnswer(JCas jcas, int begin, int end) {
    super(jcas);
    setBegin(begin);
    setEnd(end);
    readObject();
  }   

  /** <!-- begin-user-doc -->
    * Write your own initialization here
    * <!-- end-user-doc -->
  @generated modifiable */
  private void readObject() {/*default - does nothing empty block */}
     
 
    
  //*--------------*
  //* Feature: questionId

  /** getter for questionId - gets 
   * @generated */
  public String getQuestionId() {
    if (FactoidAnswer_Type.featOkTst && ((FactoidAnswer_Type)jcasType).casFeat_questionId == null)
      jcasType.jcas.throwFeatMissing("questionId", "edu.cmu.lti.oaqa.knn4qa.types.FactoidAnswer");
    return jcasType.ll_cas.ll_getStringValue(addr, ((FactoidAnswer_Type)jcasType).casFeatCode_questionId);}
    
  /** setter for questionId - sets  
   * @generated */
  public void setQuestionId(String v) {
    if (FactoidAnswer_Type.featOkTst && ((FactoidAnswer_Type)jcasType).casFeat_questionId == null)
      jcasType.jcas.throwFeatMissing("questionId", "edu.cmu.lti.oaqa.knn4qa.types.FactoidAnswer");
    jcasType.ll_cas.ll_setStringValue(addr, ((FactoidAnswer_Type)jcasType).casFeatCode_questionId, v);}    
  }

    