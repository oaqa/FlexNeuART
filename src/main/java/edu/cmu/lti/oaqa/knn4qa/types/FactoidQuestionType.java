

/* First created by JCasGen Thu Jan 12 23:50:48 EST 2017 */
package edu.cmu.lti.oaqa.knn4qa.types;

import org.apache.uima.jcas.JCas; 
import org.apache.uima.jcas.JCasRegistry;
import org.apache.uima.jcas.cas.TOP_Type;

import org.apache.uima.jcas.tcas.Annotation;


/** A question type for the factoid question
 * Updated by JCasGen Tue Jan 17 12:22:14 EST 2017
 * XML source: /home/leo/SourceTreeGit/knn4qa_oqaqa/src/main/resources/types/typeSystemDescriptor.xml
 * @generated */
public class FactoidQuestionType extends Annotation {
  /** @generated
   * @ordered 
   */
  @SuppressWarnings ("hiding")
  public final static int typeIndexID = JCasRegistry.register(FactoidQuestionType.class);
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
  protected FactoidQuestionType() {/* intentionally empty block */}
    
  /** Internal - constructor used by generator 
   * @generated */
  public FactoidQuestionType(int addr, TOP_Type type) {
    super(addr, type);
    readObject();
  }
  
  /** @generated */
  public FactoidQuestionType(JCas jcas) {
    super(jcas);
    readObject();   
  } 

  /** @generated */  
  public FactoidQuestionType(JCas jcas, int begin, int end) {
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
  //* Feature: value

  /** getter for value - gets 
   * @generated */
  public String getValue() {
    if (FactoidQuestionType_Type.featOkTst && ((FactoidQuestionType_Type)jcasType).casFeat_value == null)
      jcasType.jcas.throwFeatMissing("value", "edu.cmu.lti.oaqa.knn4qa.types.FactoidQuestionType");
    return jcasType.ll_cas.ll_getStringValue(addr, ((FactoidQuestionType_Type)jcasType).casFeatCode_value);}
    
  /** setter for value - sets  
   * @generated */
  public void setValue(String v) {
    if (FactoidQuestionType_Type.featOkTst && ((FactoidQuestionType_Type)jcasType).casFeat_value == null)
      jcasType.jcas.throwFeatMissing("value", "edu.cmu.lti.oaqa.knn4qa.types.FactoidQuestionType");
    jcasType.ll_cas.ll_setStringValue(addr, ((FactoidQuestionType_Type)jcasType).casFeatCode_value, v);}    
  }

    