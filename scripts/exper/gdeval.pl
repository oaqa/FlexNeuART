#!/usr/bin/perl -w

# Graded relevance assessment script for the TREC 2010 Web track
# Evalution measures are written to standard output in CSV format.
# 
# Currently reports only NDCG and ERR
# (see http://learningtorankchallenge.yahoo.com/instructions.php)

$usage = "usage: $0 qrels run";
$version = "version 1.2 (Tue Oct 12 15:58:46 EDT 2010)";

$MAX_JUDGMENT = 4; # Maximum gain value allowed in qrels file.
$K = 20;           # Reporting depth for results.

if ($#ARGV >= 0 && ($ARGV[0] eq "-v" || $ARGV[0] eq "-version")) {
  print "$0: $version\n";
  exit 0;
}

if ($#ARGV >= 0 && $ARGV[0] eq "-help") {
  print "$usage\n";
  exit 0;
}

die $usage unless $#ARGV == 1;
$QRELS = $ARGV[0];
$RUN = $ARGV[1];

# Read qrels file, check format, and sort
open (QRELS) || die "$0: cannot open \"$QRELS\": !$\n";
$ln=0;
while (<QRELS>) {
  $ln++;
  s/[\r\n]//g;
  ($topic, $q0, $docno, $judgment) = split (' ');
  $q0 = 0;
  die "$0: QREL format error on line $ln of \"$QRELS\"\n"
    unless $q0 == $q0 && $judgment =~ /^-?[0-9]+$/ && $judgment <= $MAX_JUDGMENT;
  if ($judgment > 0) {
    $qrels[$#qrels + 1]= "$topic $docno $judgment";
    $seen{$topic} = 1;
  }
}
close (QRELS);
@qrels = sort qrelsOrder (@qrels);

$topics = 0;
$runid = "?????";
# Read run rile, check format, and sort
open (RUN) || die "$0: cannot open \"$RUN\": !$\n";
$ln=0;
while (<RUN>) {
  $ln++;
  s/[\r\n]//g;
  ($topic, $q0, $docno, $rank, $score, $runid) = split (' ');
  $q0 = 0;
  die "$0: RUN format error on line $ln of \"$RUN\"\n"
    unless $q0 == $q0 && $rank =~ /^[0-9]+$/ && $runid;
  $run[$#run + 1] = "$topic $docno $score";
}

@run = sort runOrder (@run);

# Process qrels: store judgments and compute ideal gains
$topicCurrent = "";
for ($i = 0; $i <= $#qrels; $i++) {
  ($topic, $docno, $judgment) = split (' ', $qrels[$i]);
  if ($topic ne $topicCurrent) {
    if ($topicCurrent ne "") {
      $ideal{$topicCurrent} = &dcg($K, @gain);
      $#gain = -1;
    }
    $topicCurrent = $topic;
  }
  next if $judgment < 0;
  $judgment{"$topic:$docno"} = $gain[$#gain + 1] = $judgment;
}
if ($topicCurrent ne "") {
  $ideal{$topicCurrent} = &dcg($K, @gain);
  $#gain = -1;
}

# Process runs: compute measures for each topic and average
print "runid,topic,ndcg\@$K,err\@$K\n";
$topicCurrent = "";
for ($i = 0; $i <= $#run; $i++) {
  ($topic, $docno, $score) = split (' ', $run[$i]);
  if ($topic ne $topicCurrent) {
    if ($topicCurrent ne "") {
      &topicDone ($runid, $topicCurrent, @gain);
      $#gain = -1;
    }
    $topicCurrent = $topic;
  }
  $j  = $judgment{"$topic:$docno"};
  $j = 0 unless $j;
  $gain[$#gain + 1] = $j;
}
if ($topicCurrent ne "") {
  &topicDone ($runid, $topicCurrent, @gain);
  $#gain = -1;
}
if ($topics > 0) {
  $ndcgAvg = $ndcgTotal/$topics;
  $errAvg = $errTotal/$topics;
  printf "$runid,amean,%.5f,%.5f\n",$ndcgAvg,$errAvg;
} else {
  print "$runid,amean,0.00000,0.0000\n";
}

exit 0;

# comparison function for qrels: by topic then judgment
sub qrelsOrder {
  my ($topicA, $docnoA, $judgmentA) = split (' ', $a);
  my ($topicB, $docnoB, $judgmentB) = split (' ', $b);

  if ($topicA lt $topicB) {
    return -1;
  } elsif ($topicA gt $topicB) {
    return 1;
  } else {
    return $judgmentB <=> $judgmentA;
  }
}

# comparison function for runs: by topic then score then docno
sub runOrder {
  my ($topicA, $docnoA, $scoreA) = split (' ', $a);
  my ($topicB, $docnoB, $scoreB) = split (' ', $b);

  if ($topicA lt $topicB) {
    return -1;
  } elsif ($topicA gt $topicB) {
    return 1;
  } elsif ($scoreA < $scoreB) {
    return 1;
  } elsif ($scoreA > $scoreB) {
    return -1;
  } elsif ($docnoA lt $docnoB) {
    return 1;
  } elsif ($docnoA gt $docnoB) {
    return -1;
  } else {
    return 0;
  }
}

# compute DCG over a sorted array of gain values, reporting at depth $k
sub dcg {
 my ($k, @gain) = @_;
 my ($i, $score) = (0, 0);

 for ($i = 0; $i <= ($k <= $#gain ? $k - 1 : $#gain); $i++) {
   $score += (2**$gain[$i] - 1)/log ($i + 2);
 }
 return $score;
}

# compute ERR over a sorted array of gain values, reporting at depth $k
sub err {
  my ($k, @gain) = @_;
  my ($i, $score, $decay, $r);

 $score = 0.0;
 $decay = 1.0;
 for ($i = 0; $i <= ($k <= $#gain ? $k - 1 : $#gain); $i++) {
   $r = (2**$gain[$i] - 1)/(2**$MAX_JUDGMENT);
   $score += $r*$decay/($i + 1);
   $decay *= (1 - $r);
 }
 return $score;
}

# compute and report information for current topic
sub topicDone {
  my ($runid, $topic, @gain) = @_;
  my($ndcg, $err) = (0, 0);
  if ($seen{$topic}) {
    $ndcg = &dcg($K, @gain)/$ideal{$topic};
    $err = &err ($K, @gain);
    $ndcgTotal += $ndcg;
    $errTotal += $err;
    $topics++;
    printf  "$runid,$topicCurrent,%.5f,%.5f\n",$ndcg,$err;
  }
}
