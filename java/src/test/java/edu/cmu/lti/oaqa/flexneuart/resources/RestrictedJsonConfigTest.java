package edu.cmu.lti.oaqa.flexneuart.resources;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import com.google.gson.JsonParser;

import edu.cmu.lti.oaqa.flexneuart.utils.StringUtils;

public class RestrictedJsonConfigTest {
  @Test
  public void test() throws Exception {
    /*
     * This is a rather simplistic test, but it targets to check the main functionality including accessing nested elements,
     * including nested key-value collections, and nested arrays.
     */
    JsonParser parser = new JsonParser();

    // Caution non-boolean JSON entries except string "true" seem to be always converted to false by GSON!!!
    
    String configText = "[ {\"k1\" : 3, \"k2\" : \"v2\", \"k_arr\" : [0, 4, 5.25], \"k_arr_bool\" : [false, \"true\", true] }, "
        + "{\"k4\" : true, \"k2\" : 3.4, \"k_dict\" : {\"k1\" : \"aaaa\", \"k2\" : \"bbbb\" }, " 
        + "\"k_dict_arr\" : [ {\"k11\" : \"aaaa_1\", \"k12\" : \"bbbb_1\" } , {\"k21\" : \"aaaa_2\", \"k22\" : \"bbbb_2\" } ]}  ]"
        + "";
    RestrictedJsonConfig p = new RestrictedJsonConfig(parser.parse(configText), "root");
    RestrictedJsonConfig[] level1 = p.getParamConfigArray();
    RestrictedJsonConfig conf1 = level1[0];
    
    assertTrue(conf1.getReqParamStr("k1").compareTo("3") == 0);
    assertEquals(conf1.getReqParamInt("k1"), 3);
    assertTrue(conf1.getReqParamStr("k2").compareTo("v2") == 0);
    assertTrue(StringUtils.joinWithSpace(conf1.getParamNestedConfig("k_arr").getParamStringArray()).compareTo("0 4 5.25") == 0);
    int tmp1[] = conf1.getParamNestedConfig("k_arr").getParamIntArray();
    
    assertEquals(tmp1.length, 3);
    assertEquals(tmp1[0], 0);
    assertEquals(tmp1[1], 4);
    assertEquals(tmp1[2], 5);
    
    float tmp2[] = conf1.getParamNestedConfig("k_arr").getParamFloatArray();

    assertEquals(tmp2.length, 3);
    assertEquals(tmp2[0], 0, 1e-5);
    assertEquals(tmp2[1], 4, 1e-5);
    assertEquals(tmp2[2], 5.25, 1e-5); 
    
    boolean tmp3[] = conf1.getParamNestedConfig("k_arr_bool").getParamBoolArray();
    assertEquals(tmp3.length, 3);
    //System.out.println(tmp3[0] + " " + tmp3[1] + " " + tmp3[2]);
    assertEquals(tmp3[0], false);
    assertEquals(tmp3[1], true);
    assertEquals(tmp3[2], true);
    
    RestrictedJsonConfig conf2 = level1[1];
    
    assertTrue(conf2.getReqParamStr("k4").compareTo("true") == 0);
    assertTrue(conf2.getReqParamStr("k2").compareTo("3.4") == 0);
    
    RestrictedJsonConfig confNested1 = conf2.getParamNestedConfig("k_dict");
    assertTrue(confNested1.getReqParamStr("k1").compareTo("aaaa") == 0);
    assertTrue(confNested1.getReqParamStr("k2").compareTo("bbbb") == 0);
    
    RestrictedJsonConfig confNested2 = conf2.getParamNestedConfig("k_dict_arr");
    RestrictedJsonConfig confNested2_1 = confNested2.getParamConfigArray()[0];
    assertTrue(confNested2_1.getReqParamStr("k11").compareTo("aaaa_1") == 0);
    assertTrue(confNested2_1.getReqParamStr("k12").compareTo("bbbb_1") == 0);
    RestrictedJsonConfig confNested2_2 = confNested2.getParamConfigArray()[1];
    assertTrue(confNested2_2.getReqParamStr("k21").compareTo("aaaa_2") == 0);
    assertTrue(confNested2_2.getReqParamStr("k22").compareTo("bbbb_2") == 0);
    
  }

}
