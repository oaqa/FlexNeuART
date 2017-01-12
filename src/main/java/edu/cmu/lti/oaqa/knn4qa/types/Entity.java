

/* First created by JCasGen Thu Jan 12 13:26:20 EST 2017 */
package edu.cmu.lti.oaqa.knn4qa.types;

import org.apache.uima.jcas.JCas; 
import org.apache.uima.jcas.JCasRegistry;
import org.apache.uima.jcas.cas.TOP_Type;

import org.apache.uima.jcas.tcas.Annotation;


/** An annotation with type & label attributes (used for NERs and DBPedia entiteies
 * Updated by JCasGen Thu Jan 12 13:26:20 EST 2017
 * XML source: /home/leo/SourceTreeGit/knn4qa_oqaqa/src/main/resources/types/typeSystemDescriptor.xml
 * @generated */
public class Entity extends Annotation {
  /** @generated
   * @ordered 
   */
  @SuppressWarnings ("hiding")
  public final static int typeIndexID = JCasRegistry.register(Entity.class);
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
  protected Entity() {/* intentionally empty block */}
    
  /** Internal - constructor used by generator 
   * @generated
   * @param addr low level Feature Structure reference
   * @param type the type of this Feature Structure 
   */
  public Entity(int addr, TOP_Type type) {
    super(addr, type);
    readObject();
  }
  
  /** @generated
   * @param jcas JCas to which this Feature Structure belongs 
   */
  public Entity(JCas jcas) {
    super(jcas);
    readObject();   
  } 

  /** @generated
   * @param jcas JCas to which this Feature Structure belongs
   * @param begin offset to the begin spot in the SofA
   * @param end offset to the end spot in the SofA 
  */  
  public Entity(JCas jcas, int begin, int end) {
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
  //* Feature: etype

  /** getter for etype - gets Entity type
   * @generated
   * @return value of the feature 
   */
  public String getEtype() {
    if (Entity_Type.featOkTst && ((Entity_Type)jcasType).casFeat_etype == null)
      jcasType.jcas.throwFeatMissing("etype", "edu.cmu.lti.oaqa.knn4qa.types.Entity");
    return jcasType.ll_cas.ll_getStringValue(addr, ((Entity_Type)jcasType).casFeatCode_etype);}
    
  /** setter for etype - sets Entity type 
   * @generated
   * @param v value to set into the feature 
   */
  public void setEtype(String v) {
    if (Entity_Type.featOkTst && ((Entity_Type)jcasType).casFeat_etype == null)
      jcasType.jcas.throwFeatMissing("etype", "edu.cmu.lti.oaqa.knn4qa.types.Entity");
    jcasType.ll_cas.ll_setStringValue(addr, ((Entity_Type)jcasType).casFeatCode_etype, v);}    
   
    
  //*--------------*
  //* Feature: label

  /** getter for label - gets Entity label
   * @generated
   * @return value of the feature 
   */
  public String getLabel() {
    if (Entity_Type.featOkTst && ((Entity_Type)jcasType).casFeat_label == null)
      jcasType.jcas.throwFeatMissing("label", "edu.cmu.lti.oaqa.knn4qa.types.Entity");
    return jcasType.ll_cas.ll_getStringValue(addr, ((Entity_Type)jcasType).casFeatCode_label);}
    
  /** setter for label - sets Entity label 
   * @generated
   * @param v value to set into the feature 
   */
  public void setLabel(String v) {
    if (Entity_Type.featOkTst && ((Entity_Type)jcasType).casFeat_label == null)
      jcasType.jcas.throwFeatMissing("label", "edu.cmu.lti.oaqa.knn4qa.types.Entity");
    jcasType.ll_cas.ll_setStringValue(addr, ((Entity_Type)jcasType).casFeatCode_label, v);}    
  }

    