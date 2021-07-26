package edu.cmu.lti.oaqa.flexneuart.fwdindx;

import java.util.concurrent.ConcurrentMap;

import org.mapdb.DB;
import org.mapdb.DBMaker;
import org.mapdb.Serializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MapDbBackend extends PersistentKeyValBackend {
  
  private static final Logger logger = LoggerFactory.getLogger(MapDbBackend.class);
 
  public static final int MEM_ALLOCATE_INCREMENT = 1024*1024*1; // Allocating memory in 16 MB chunks
  public static final int MAX_NODE_SIZE = 32;
  
  @Override
  public int size() {
    // requires counterEnable
    return mDbMap.size();
  }

  @Override
  public void openIndexForReading(String indexPrefix) throws Exception {
    // Note that we disable file locking and concurrence to enable accessing the file by different threads at the same time
    mDb = DBMaker.fileDB(indexPrefix)
                .allocateIncrement(MEM_ALLOCATE_INCREMENT)
                .concurrencyDisable()
                .fileLockDisable()
                .closeOnJvmShutdown()
                .fileMmapEnable()
                .readOnly()
                .make();
    
    mDbMap = mDb.hashMap("map", Serializer.STRING, Serializer.BYTE_ARRAY)
                  // enabling counters is crucial for the size function
                  .counterEnable()
                  .open();

    logger.info("MapDB opened for reading: " + indexPrefix);
  }

  @Override
  public void initIndexForWriting(String indexPrefix, int expectedQty) throws Exception {
    /* 
     * According to https://jankotek.gitbooks.io/mapdb/content/htreemap/
     * Maximal Hash Table Size is calculated as: segment# * node size ^ level count
     * So, the settings below give us approximately 134M : 4 * 32^5 
     * 
     */
    int levelQty = Math.max(1, (int) Math.floor(Math.log(expectedQty) / Math.log(MAX_NODE_SIZE)));
    int segmentQty =  Math.max(1, (int) Math.ceil(expectedQty / Math.pow(MAX_NODE_SIZE, levelQty)));
    
    mDb = DBMaker.fileDB(indexPrefix)
        .allocateIncrement(MEM_ALLOCATE_INCREMENT)
        .closeOnJvmShutdown()
        .fileMmapEnable().make();
    
    // With respect to layout see https://jankotek.gitbooks.io/mapdb/content/htreemap/
    mDbMap = mDb.hashMap("map", Serializer.STRING, Serializer.BYTE_ARRAY)
                .layout(segmentQty, MAX_NODE_SIZE, levelQty)
                // enabling counters is crucial for the size function
                .counterEnable()
                .create();
    
    logger.info("MapDB opened for writing: " + indexPrefix + " using expected # of entries: " + expectedQty);
    logger.info(String.format("# of segments/block size/# of levels %d/%d/%d", segmentQty, MAX_NODE_SIZE, levelQty));
  }

  @Override
  public void put(String key, byte[] value) {
    mDbMap.put(key, value);   
  }

  @Override
  public byte[] get(String key) {
    return mDbMap.get(key);
  }

  @Override
  public String[] getKeyArray() {
    String res[] = new String[0];
    return mDbMap.keySet().toArray(res);
  }
  
  @Override
  public void close() {
    if (mDb != null) {
      mDb.commit();
      mDb.close();
      mDb = null;
      mDbMap = null;
    }
  }

  private DB mDb;
  ConcurrentMap<String, byte[]> mDbMap;
}
