

/* First created by JCasGen Thu Jan 12 13:26:20 EST 2017 */
package edu.cmu.lti.oaqa.knn4qa.types;

import org.apache.uima.jcas.JCas; 
import org.apache.uima.jcas.JCasRegistry;
import org.apache.uima.jcas.cas.TOP_Type;

import org.apache.uima.jcas.tcas.Annotation;


/** 
 * Updated by JCasGen Thu Jan 12 23:52:10 EST 2017
 * XML source: /home/leo/SourceTreeGit/knn4qa_oqaqa/src/main/resources/types/typeSystemDescriptor.xml
 * @generated */
public class TokenLemma extends Annotation {
  /** @generated
   * @ordered 
   */
  @SuppressWarnings ("hiding")
  public final static int typeIndexID = JCasRegistry.register(TokenLemma.class);
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
  protected TokenLemma() {/* intentionally empty block */}
    
  /** Internal - constructor used by generator 
   * @generated
   * @param addr low level Feature Structure reference
   * @param type the type of this Feature Structure 
   */
  public TokenLemma(int addr, TOP_Type type) {
    super(addr, type);
    readObject();
  }
  
  /** @generated
   * @param jcas JCas to which this Feature Structure belongs 
   */
  public TokenLemma(JCas jcas) {
    super(jcas);
    readObject();   
  } 

  /** @generated
   * @param jcas JCas to which this Feature Structure belongs
   * @param begin offset to the begin spot in the SofA
   * @param end offset to the end spot in the SofA 
  */  
  public TokenLemma(JCas jcas, int begin, int end) {
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
  //* Feature: Lemma

  /** getter for Lemma - gets 
   * @generated */
  public String getLemma() {
    if (TokenLemma_Type.featOkTst && ((TokenLemma_Type)jcasType).casFeat_Lemma == null)
      jcasType.jcas.throwFeatMissing("Lemma", "edu.cmu.lti.oaqa.knn4qa.types.TokenLemma");
    return jcasType.ll_cas.ll_getStringValue(addr, ((TokenLemma_Type)jcasType).casFeatCode_Lemma);}
    
  /** setter for Lemma - sets  
   * @generated */
  public void setLemma(String v) {
    if (TokenLemma_Type.featOkTst && ((TokenLemma_Type)jcasType).casFeat_Lemma == null)
      jcasType.jcas.throwFeatMissing("Lemma", "edu.cmu.lti.oaqa.knn4qa.types.TokenLemma");
    jcasType.ll_cas.ll_setStringValue(addr, ((TokenLemma_Type)jcasType).casFeatCode_Lemma, v);}    
  }

    