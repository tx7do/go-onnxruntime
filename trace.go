package onnxruntime

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"time"

	"github.com/c3sr/tracer"
	"github.com/opentracing/opentracing-go"
)

type TraceEvent struct {
	Category  string            `json:"cat,omitempty"`
	Name      string            `json:"name,omitempty"`
	Phase     string            `json:"ph,omitempty"`
	Timestamp int64             `json:"ts,omitempty"`
	Duration  int64             `json:"dur,omitempty"`
	ProcessID int64             `json:"pid,omitempty"`
	ThreadID  int64             `json:"tid,omitempty"`
	Arguments map[string]string `json:"args,omitempty"`
	Start     int64             `json:"-"`
	End       int64             `json:"-"`
	StartTime time.Time         `json:"-"`
	EndTime   time.Time         `json:"-"`
}

func (e *TraceEvent) ID() string {
	return fmt.Sprintf("%s/%v", e.Name, e.ThreadID)
}

type TraceEvents []TraceEvent

func (t TraceEvents) Len() int           { return len(t) }
func (t TraceEvents) Swap(i, j int)      { t[i], t[j] = t[j], t[i] }
func (t TraceEvents) Less(i, j int) bool { return t[i].Start < t[j].Start }

type Trace struct {
	StartTime   time.Time
	TraceEvents TraceEvents
}

func (t *Trace) Len() int           { return t.TraceEvents.Len() }
func (t *Trace) Swap(i, j int)      { t.TraceEvents.Swap(i, j) }
func (t *Trace) Less(i, j int) bool { return t.TraceEvents.Less(i, j) }

func SplitTrace(t *Trace, startSlice []int64, endSlice []int64) ([]*Trace, error) {
	batchNum := 0
	var tSlice []*Trace
	tmpTrace := new(Trace)
	tSlice = append(tSlice, tmpTrace)
	tmpTrace.StartTime = time.Unix(0, startSlice[batchNum])
	for _, event := range t.TraceEvents {
		if event.End < endSlice[batchNum] {
			if event.Start > startSlice[batchNum] {
				tmpTrace.TraceEvents = append(tmpTrace.TraceEvents, event)
			}
		} else {
			batchNum++
			tmpTrace = new(Trace)
			tSlice = append(tSlice, tmpTrace)
			tmpTrace.StartTime = time.Unix(0, startSlice[batchNum])
			tmpTrace.TraceEvents = append(tmpTrace.TraceEvents, event)
		}
	}
	return tSlice, nil
}

func NewTrace(data string, startTime int64) (*Trace, error) {
	trace := new(Trace)
	if err := json.Unmarshal([]byte(data), &trace.TraceEvents); err != nil {
		return nil, err
	}
	trace.StartTime = time.Unix(0, startTime)
	for i, event := range trace.TraceEvents {
		trace.TraceEvents[i].Start = startTime + event.Timestamp*1000
		trace.TraceEvents[i].StartTime = time.Unix(0, trace.TraceEvents[i].Start)
		trace.TraceEvents[i].End = startTime + event.Timestamp*1000 + event.Duration*1000
		trace.TraceEvents[i].EndTime = time.Unix(0, trace.TraceEvents[i].End)
	}

	sort.Sort(trace.TraceEvents)

	return trace, nil
}

func (e *TraceEvent) Publish(ctx context.Context, lvl tracer.Level, _ ...opentracing.StartSpanOption) error {
	// skip the events for loading model and setting environments in onnxruntime
	if e.Name == "model_loading_from_uri" || e.Name == "session_initialization" || e.Name == "model_run" || e.Name == "SequentialExecutor::Execute" {
		return nil
	}

	tags := opentracing.Tags{
		"category":   e.Category,
		"phase":      e.Phase,
		"process_id": e.ProcessID,
		"thread_id":  e.ThreadID,
		"arguments":  e.Arguments,
	}
	s, _ := tracer.StartSpanFromContext(
		ctx,
		lvl,
		e.Name,
		opentracing.StartTime(e.StartTime),
		tags,
	)
	if s == nil {
		log.WithField("event_name", e.Name).
			WithField("tags", tags).
			Error("failed to create span from context")
		return nil
	}
	s.FinishWithOptions(opentracing.FinishOptions{
		FinishTime: e.EndTime,
	})
	return nil
}

func (t *Trace) Publish(ctx context.Context, lvl tracer.Level, opts ...opentracing.StartSpanOption) error {
	for _, event := range t.TraceEvents {
		if err := event.Publish(ctx, lvl, opts...); err != nil {
			return err
		}
	}
	return nil
}
