

/* First created by JCasGen Thu Jan 12 21:27:33 EST 2017 */
package edu.cmu.lti.oaqa.knn4qa.types;

import org.apache.uima.jcas.JCas; 
import org.apache.uima.jcas.JCasRegistry;
import org.apache.uima.jcas.cas.TOP_Type;

import org.apache.uima.jcas.tcas.Annotation;


/** w-words: 
WDT	Wh-determiner
WP	        Wh-pronoun,
WP$	Possessive wh-pronoun
WRB	Wh-adverb
 * Updated by JCasGen Thu Jan 12 23:52:10 EST 2017
 * XML source: /home/leo/SourceTreeGit/knn4qa_oqaqa/src/main/resources/types/typeSystemDescriptor.xml
 * @generated */
public class WWord extends Annotation {
  /** @generated
   * @ordered 
   */
  @SuppressWarnings ("hiding")
  public final static int typeIndexID = JCasRegistry.register(WWord.class);
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
  protected WWord() {/* intentionally empty block */}
    
  /** Internal - constructor used by generator 
   * @generated */
  public WWord(int addr, TOP_Type type) {
    super(addr, type);
    readObject();
  }
  
  /** @generated */
  public WWord(JCas jcas) {
    super(jcas);
    readObject();   
  } 

  /** @generated */  
  public WWord(JCas jcas, int begin, int end) {
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
    if (WWord_Type.featOkTst && ((WWord_Type)jcasType).casFeat_value == null)
      jcasType.jcas.throwFeatMissing("value", "edu.cmu.lti.oaqa.knn4qa.types.WWord");
    return jcasType.ll_cas.ll_getStringValue(addr, ((WWord_Type)jcasType).casFeatCode_value);}
    
  /** setter for value - sets  
   * @generated */
  public void setValue(String v) {
    if (WWord_Type.featOkTst && ((WWord_Type)jcasType).casFeat_value == null)
      jcasType.jcas.throwFeatMissing("value", "edu.cmu.lti.oaqa.knn4qa.types.WWord");
    jcasType.ll_cas.ll_setStringValue(addr, ((WWord_Type)jcasType).casFeatCode_value, v);}    
  }

    