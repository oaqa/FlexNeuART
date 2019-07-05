/*
 *  Copyright 2018+ Carnegie Mellon University
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

package edu.cmu.lti.oaqa.knn4qa.utils;

import java.net.URI;
import java.net.URLDecoder;
import java.util.HashSet;
import java.util.HashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.htmlparser.Tag;
import org.htmlparser.Text;
import org.htmlparser.util.Translate;
import org.htmlparser.visitors.NodeVisitor;

class Pair<T, U> {
  private final T first;
  private final U second;

  public Pair(T f, U s) {
    this.first = f;
    this.second = s;
  }

  public T getFirst() {
    return first;
  }

  public U getSecond() {
    return second;
  }
};

/**
 * HTML-parser/cleaner used in TREC 19,20,21 adhoc.
 * 
 * @author Leonid Boytsov
 *
 */
class CleanerUtil extends NodeVisitor {
  public CleanerUtil(String BaseHref) {
    SetBaseHref(BaseHref);
    mTextBuffer = new StringBuilder();
    mHrefBuffer = new StringBuilder();
    mHrefSnippets = new HashMap<String, String>();

    if (mParaTags == null) {
      InitParaTags();
    }
  }

  static public String CollapseSpaces(String s) {
    s = Replace(s, "\\r", "\n");
    // Looks like quantifiers +, * here may lead to an infinite recursion
    return Replace(s, "[\\t ]{1,10}", " ");
  }

  public String GetBodyText() {
    return CollapseSpaces(mTextBuffer.toString());
  }

  public String GetLinkText() {
    return CollapseSpaces(mHrefBuffer.toString());
  }

  public String GetTitleText() {
    return CollapseSpaces(mTitle);
  }

  public String GetKeywordText() {
    return CollapseSpaces(mKeywords);
  }

  public String GetDescriptionText() {
    return CollapseSpaces(mDescription);
  }

  public HashMap<String, String> GetHrefSnippets() {
    return mHrefSnippets;
  }

  public void visitTag(Tag tag) {
    String Name = tag.getTagName().toLowerCase();

    // System.out.println(Name + " -> " + tag.isEndTag());
    String attr, content;

    if (Name.equals("meta")) {
      if ((attr = tag.getAttribute("name")) != null) {
        boolean bDesc = attr.equals("description");
        boolean bKeyw = attr.equals("keywords");

        if ((bDesc || bKeyw) && (content = tag.getAttribute("content")) != null) {
          content = CleanText(content);
          if (bDesc)
            mDescription += content;
          if (bKeyw)
            mKeywords += content;
        }
      }
    } else if (Name.equals("base") && (content = tag.getAttribute("href")) != null) {
      SetBaseHref(content);
    } else if ((Name.equals("frame") || Name.equals("iframe")) && (content = tag.getAttribute("src")) != null) {
      AddLinkOut(content, "");
    }

    SetFlags(Name, true);

    if (mInHref) {
      mHrefAddr = tag.getAttribute("href");
      mLinkText = "";
    }

    ProcessParagraphs(Name, true);
  }

  private void AddLinkOut(String HrefAddr, String LinkText) {
    String ResolvedAddr = ResolveURL(HrefAddr);

    if (LinkText == null) {
      LinkText = "";
    }

    if (ResolvedAddr != null) {
      String OldText = mHrefSnippets.get(ResolvedAddr);

      /* More than one link to the same URL can be present on a page */
      if (OldText != null) {
        // System.out.println(ResolvedAddr + "@@ " + OldText + " >" + LinkText + "<");
        OldText = OldText + " " + LinkText;
      } else {
        OldText = LinkText;
      }

      mHrefSnippets.put(ResolvedAddr, OldText);
    }
    mHrefBuffer.append(LinkText);
  }

  public void visitEndTag(Tag tag) {
    String Name = tag.getTagName().toLowerCase();

    if (mInHref && mHrefAddr != null) {
      AddLinkOut(mHrefAddr, mLinkText);
    }

    // System.out.println(Name + " -> " + tag.isEndTag());

    SetFlags(Name, false);
    if (Name.equals("head")) {
      mInBody = true;
    }
    ProcessParagraphs(Name, false);
  }

  private void ProcessParagraphs(String Name, boolean IsStart) {
    if (mParaTags.contains(Name)) {
      mTextBuffer.append("\n");
      mLinkText += " ";

      if (!mInBody) {
        mInBody = true;
      }
    }
  }

  private void SetFlags(String Name, boolean val) {
    if (Name.equals("title")) {
      mInTitle = val;
    } else if (Name.equals("body")) {
      mInBody = val;
    } else if (Name.equals("a")) {
      mInHref = val;
    } else if (Name.equals("script")) {
      mInScript = val;
    } else if (Name.equals("style")) {
      mInStyle = val;
    }
  }

  public static String Replace(String src, String pat, String repl, boolean bIgnoreCase) {
    Pattern p = bIgnoreCase ? Pattern.compile(pat, Pattern.CASE_INSENSITIVE | Pattern.MULTILINE | Pattern.DOTALL)
        : Pattern.compile(pat, Pattern.MULTILINE | Pattern.DOTALL);
    Matcher m = p.matcher(src);

    return m.replaceAll(repl);
  }

  // Case insensitive searching.
  public static String Replace(String src, String pat, String repl) {
    return Replace(src, pat, repl, true);
  }

  public void visitStringNode(Text TextNode) {
    if (!mInScript && !mInStyle) {
      String text = CleanText(TextNode.getText());

      if (!text.isEmpty()) {
        // System.out.println(mCurrentTag + "->" + text);

        if (mInTitle) {
          mTitle += text;
          mTitle += " ";
        } else if (mInBody) {
          mTextBuffer.append(text);
          if (mInHref) {
            mLinkText += text;
          }
        }
      }
    }
  }

  private static String replaceNonBreakingSpaceWithOrdinarySpace(String text) {
    return text.replace('\u00a0', ' ');
  }

  private static String CleanText(String text) {
    return replaceNonBreakingSpaceWithOrdinarySpace(Translate.decode(text));
  }

  private void InitParaTags() {
    mParaTags = new HashSet<String>();
    // Paragraph elements
    mParaTags.add("p");
    mParaTags.add("div");
    mParaTags.add("br");
    mParaTags.add("hr");
    mParaTags.add("h1");
    mParaTags.add("h2");
    mParaTags.add("h3");
    mParaTags.add("h4");
    mParaTags.add("h5");
    mParaTags.add("h6");
    mParaTags.add("h7");
    // Misc
    mParaTags.add("a");
    mParaTags.add("pre");
    mParaTags.add("blockquote");
    mParaTags.add("q");
    mParaTags.add("address");
    mParaTags.add("dir");
    mParaTags.add("dd");
    mParaTags.add("dl");
    mParaTags.add("dt");
    mParaTags.add("menu");
    mParaTags.add("noframes");
    mParaTags.add("noscript");
    // It's better to include both in case someone forgets them in an HTML page
    mParaTags.add("body");
    mParaTags.add("head");
    // Frames
    mParaTags.add("frame");
    mParaTags.add("iframe");
    mParaTags.add("frameset");
    // Input elements
    mParaTags.add("input");
    mParaTags.add("form");
    mParaTags.add("textarea");
    mParaTags.add("button");
    mParaTags.add("select");
    mParaTags.add("option");
    mParaTags.add("optgroup");
    // List elemens
    mParaTags.add("ol");
    mParaTags.add("ul");
    mParaTags.add("li");
    // Table elements
    mParaTags.add("tr");
    mParaTags.add("td");
    mParaTags.add("table");
    mParaTags.add("tbody");
    mParaTags.add("thead");
    mParaTags.add("tfoot");
    mParaTags.add("th");
  }

  public static String NormalizeURL(String URL) {
    URI uri;
    try {
      uri = new URI(URL);
    } catch (Exception e) {
      return URL.trim();
    }
    String host = uri.getHost();
    String scheme = uri.getScheme();

    if (host == null || scheme == null
        || (!scheme.equals("http") && !scheme.equals("https") && !scheme.equals("ftp"))) {
      return URL.trim();
    }

    String Path = uri.getPath();

    if (Path == null || Path.isEmpty()) {
      Path = "/";
    }

    try {
      uri = new URI(scheme, null /* user info */, host, uri.getPort(), Path, null /* query */, null /* fragment */);
    } catch (Exception e) {
      return URL.trim();
    }
    ;

    return uri.toString().trim();
  }

  static public String GetURLWords(String URL, String Charset) {
    try {

      URI uri = new URI(URL);
      String URLWords = uri.getHost();
      if (URLWords == null) {
        URLWords = "";
      }
      if (uri.getPath() != null) {
        URLWords += " " + uri.getPath();
      }
      try {
        URLWords = URLDecoder.decode(URLWords, Charset);
      } catch (Exception e) {
      }
      URLWords = CleanerUtil.Replace(URLWords, "[.]php[0-9]?$", "");
      URLWords = CleanerUtil.Replace(URLWords, "[.]asp$", "");
      URLWords = CleanerUtil.Replace(URLWords, "[.]x?html?$", "");
      URLWords = CleanerUtil.Replace(URLWords, "[-=?_/#,.+%]", " ");

      return URLWords;
    } catch (Exception e) {

    }
    return "";
  }

  private String ResolveURL(String URL) {
    String NormURL = NormalizeURL(URL);

    try {
      // resolve relative paths
      return NormalizeURL((mBaseURI.resolve(NormURL)).toString());
    } catch (java.lang.IllegalArgumentException iaEx) {
    } catch (Exception e) {
    }

    return NormURL;
  }

  private void SetBaseHref(String BaseHref) {
    mBaseURI = null;

    if (BaseHref == null) {
      return;
    }

    try {
      mBaseURI = new URI(NormalizeURL(BaseHref));
    } catch (Exception e) {
    }
  }

  private StringBuilder mTextBuffer;
  private StringBuilder mHrefBuffer;
  private String mTitle = "";
  private String mDescription = "";
  private String mKeywords = "";
  private String mHrefAddr;
  private String mLinkText;

  private URI mBaseURI = null;

  private boolean mInBody = false;
  private boolean mInTitle = false;
  private boolean mInScript = false;
  private boolean mInHref = false;
  private boolean mInStyle = false;

  private HashMap<String, String> mHrefSnippets;

  private HashSet<String> mParaTags;

  public static Pair<String, String> SimpleProc(String text) {
    String Body = CleanText(text);
    String Title = "";

    Pattern p = Pattern.compile("<title>(.*?)<[/]title>",
        Pattern.CASE_INSENSITIVE | Pattern.MULTILINE | Pattern.DOTALL);
    Matcher m = p.matcher(Body);

    if (m.find()) {
      Title = m.group(1);
    }

    Body = Replace(Body, "<script[^>]*>.*?<[/]script>", "", true); // Remove all script tags
    Body = Replace(Body, "<style[^>]*>.*?<[/]style>", "", true); // Remove all content inside style
    Body = Replace(Body, "<!\\-\\-.*?\\-\\->", "");
    Body = Replace(Body, "<[^>]*?>", " "); // Remove all tags

    return new Pair<String, String>(CollapseSpaces(Title), CollapseSpaces(Body));
  }
};
