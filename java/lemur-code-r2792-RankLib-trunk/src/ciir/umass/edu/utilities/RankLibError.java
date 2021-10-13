package ciir.umass.edu.utilities;

/**
 * Instead of using random error types, use RankLibError exceptions throughout
 *   -- this means that clients can catch-all from us easily.
 * @author jfoley
 */
public class RankLibError extends RuntimeException {
  private RankLibError(Exception e) { super(e); }
  private RankLibError(String message) {
    super(message);
  }
  private RankLibError(String message, Exception cause) {
    super(message, cause);
  }

  /** Don't rewrap RankLibErrors in RankLibErrors */
  public static RankLibError create(Exception e) {
    if(e instanceof RankLibError) {
      return (RankLibError) e;
    }
    return new RankLibError(e);
  }

  public static RankLibError create(String message) {
    return new RankLibError(message);
  }

  /** Don't rewrap RankLibErrors in RankLibErrors */
  public static RankLibError create(String message, Exception cause) {
    if(cause instanceof RankLibError) {
      return (RankLibError) cause;
    }
    return new RankLibError(message, cause);
  }
}
