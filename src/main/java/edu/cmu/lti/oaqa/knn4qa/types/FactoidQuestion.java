

/* First created by JCasGen Thu Jan 12 13:26:20 EST 2017 */
package edu.cmu.lti.oaqa.knn4qa.types;

import org.apache.uima.jcas.JCas; 
import org.apache.uima.jcas.JCasRegistry;
import org.apache.uima.jcas.cas.TOP_Type;

import org.apache.uima.jcas.cas.FSArray;
import org.apache.uima.jcas.tcas.Annotation;


/** A factoid question
 * Updated by JCasGen Tue Jan 17 12:22:14 EST 2017
 * XML source: /home/leo/SourceTreeGit/knn4qa_oqaqa/src/main/resources/types/typeSystemDescriptor.xml
 * @generated */
public class FactoidQuestion extends Annotation {
  /** @generated
   * @ordered 
   */
  @SuppressWarnings ("hiding")
  public final static int typeIndexID = JCasRegistry.register(FactoidQuestion.class);
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
  protected FactoidQuestion() {/* intentionally empty block */}
    
  /** Internal - constructor used by generator 
   * @generated
   * @param addr low level Feature Structure reference
   * @param type the type of this Feature Structure 
   */
  public FactoidQuestion(int addr, TOP_Type type) {
    super(addr, type);
    readObject();
  }
  
  /** @generated
   * @param jcas JCas to which this Feature Structure belongs 
   */
  public FactoidQuestion(JCas jcas) {
    super(jcas);
    readObject();   
  } 

  /** @generated
   * @param jcas JCas to which this Feature Structure belongs
   * @param begin offset to the begin spot in the SofA
   * @param end offset to the end spot in the SofA 
  */  
  public FactoidQuestion(JCas jcas, int begin, int end) {
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
  //* Feature: id

  /** getter for id - gets 
   * @generated */
  public String getId() {
    if (FactoidQuestion_Type.featOkTst && ((FactoidQuestion_Type)jcasType).casFeat_id == null)
      jcasType.jcas.throwFeatMissing("id", "edu.cmu.lti.oaqa.knn4qa.types.FactoidQuestion");
    return jcasType.ll_cas.ll_getStringValue(addr, ((FactoidQuestion_Type)jcasType).casFeatCode_id);}
    
  /** setter for id - sets  
   * @generated */
  public void setId(String v) {
    if (FactoidQuestion_Type.featOkTst && ((FactoidQuestion_Type)jcasType).casFeat_id == null)
      jcasType.jcas.throwFeatMissing("id", "edu.cmu.lti.oaqa.knn4qa.types.FactoidQuestion");
    jcasType.ll_cas.ll_setStringValue(addr, ((FactoidQuestion_Type)jcasType).casFeatCode_id, v);}    
   
    
  //*--------------*
  //* Feature: answers

  /** getter for answers - gets 
   * @generated */
  public FSArray getAnswers() {
    if (FactoidQuestion_Type.featOkTst && ((FactoidQuestion_Type)jcasType).casFeat_answers == null)
      jcasType.jcas.throwFeatMissing("answers", "edu.cmu.lti.oaqa.knn4qa.types.FactoidQuestion");
    return (FSArray)(jcasType.ll_cas.ll_getFSForRef(jcasType.ll_cas.ll_getRefValue(addr, ((FactoidQuestion_Type)jcasType).casFeatCode_answers)));}
    
  /** setter for answers - sets  
   * @generated */
  public void setAnswers(FSArray v) {
    if (FactoidQuestion_Type.featOkTst && ((FactoidQuestion_Type)jcasType).casFeat_answers == null)
      jcasType.jcas.throwFeatMissing("answers", "edu.cmu.lti.oaqa.knn4qa.types.FactoidQuestion");
    jcasType.ll_cas.ll_setRefValue(addr, ((FactoidQuestion_Type)jcasType).casFeatCode_answers, jcasType.ll_cas.ll_getFSRef(v));}    
    
  /** indexed getter for answers - gets an indexed value - 
   * @generated */
  public FactoidAnswer getAnswers(int i) {
    if (FactoidQuestion_Type.featOkTst && ((FactoidQuestion_Type)jcasType).casFeat_answers == null)
      jcasType.jcas.throwFeatMissing("answers", "edu.cmu.lti.oaqa.knn4qa.types.FactoidQuestion");
    jcasType.jcas.checkArrayBounds(jcasType.ll_cas.ll_getRefValue(addr, ((FactoidQuestion_Type)jcasType).casFeatCode_answers), i);
    return (FactoidAnswer)(jcasType.ll_cas.ll_getFSForRef(jcasType.ll_cas.ll_getRefArrayValue(jcasType.ll_cas.ll_getRefValue(addr, ((FactoidQuestion_Type)jcasType).casFeatCode_answers), i)));}

  /** indexed setter for answers - sets an indexed value - 
   * @generated */
  public void setAnswers(int i, FactoidAnswer v) { 
    if (FactoidQuestion_Type.featOkTst && ((FactoidQuestion_Type)jcasType).casFeat_answers == null)
      jcasType.jcas.throwFeatMissing("answers", "edu.cmu.lti.oaqa.knn4qa.types.FactoidQuestion");
    jcasType.jcas.checkArrayBounds(jcasType.ll_cas.ll_getRefValue(addr, ((FactoidQuestion_Type)jcasType).casFeatCode_answers), i);
    jcasType.ll_cas.ll_setRefArrayValue(jcasType.ll_cas.ll_getRefValue(addr, ((FactoidQuestion_Type)jcasType).casFeatCode_answers), i, jcasType.ll_cas.ll_getFSRef(v));}
  }

    