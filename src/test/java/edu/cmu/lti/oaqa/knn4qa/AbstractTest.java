package edu.cmu.lti.oaqa.knn4qa;

import org.slf4j.Logger;

public class AbstractTest {

  public AbstractTest() {
    super();
  }

  protected boolean approxEqual(Logger logger, double f1, double f2, double threshold) {
    boolean res = Math.abs(f1 - f2) < threshold;
    logger.info(String.format("Comparing %f vs %f with threshold %f, result %b",
                              f1, f2, threshold, res));
    return res;
  }

}
